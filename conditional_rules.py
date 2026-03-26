"""Conditional rules pipeline: reasoning with interacting, branching rules.

The reasoning is genuinely hard because rules interact:
  - Conditional rules activate based on measurements
  - Different images trigger different rule combinations
  - The correct reasoning path is image-dependent

Levels:
  1: 2 independent rules (diameter + spacing). SFT handles.
  2: 3 rules, 1 conditional (big holes need more spacing). 
  3: 4 rules, 2 conditionals + aggregate (area limit).
  4: 5 rules, chained conditionals (area + big hole = critical).

Output format (two-stage):
  [measurements - perception, answer-only]
  [rule checking - reasoning with conditionals]
  [total violation]

Pipeline:
  --generate           Shortcut-proof dataset, all levels
  --sft                SFT on levels 1-2 (teach format + simple reasoning)
  --eval sft           Evaluate on all levels
  --bin                pass@8 difficulty binning
  --grpo               GRPO on medium bin
  --eval grpo          Evaluate on all levels
  --summary            Compare SFT vs GRPO per level

Usage:
    python3 conditional_rules.py --generate
    python3 conditional_rules.py --sft
    python3 conditional_rules.py --eval sft
    python3 conditional_rules.py --bin
    python3 conditional_rules.py --grpo
    python3 conditional_rules.py --eval grpo
    python3 conditional_rules.py --summary
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from PIL import Image
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"
WARMSTART_CKPT: str = "checkpoints_task1_sft3/final"
DATASET_DIR: str = "dataset_conditional"
LORA_RANK: int = 64
LORA_ALPHA: int = 128
LEARNING_RATE: float = 5e-6
NUM_EPOCHS_SFT: int = 3
LOG_EVERY: int = 10

NUM_GENERATIONS: int = 4
MAX_NEW_TOKENS: int = 512
KL_BETA: float = 0.05
TAU_POS: float = 1.0
TAU_NEG: float = 1.05

EASY_THRESHOLD: float = 0.75
MEDIUM_LOW: float = 0.1

SCALE_BAR_VALUES: list[int] = [5, 10, 15, 20, 25, 30, 40, 50]


# ========================
# RULE ENGINE
# ========================

def make_rules(rng: np.random.Generator, level: int) -> dict[str, Any]:
    """Generate a rule set for the given difficulty level."""
    diam_min = round(float(rng.uniform(8, 13)), 1)
    diam_max = round(diam_min + float(rng.uniform(3, 7)), 1)
    default_spacing = round(float(rng.uniform(15, 22)), 1)

    rules: dict[str, Any] = {
        "level": level,
        "diam_min": diam_min,
        "diam_max": diam_max,
        "default_spacing": default_spacing,
    }

    if level >= 2:
        # Conditional: big holes need more spacing
        rules["big_threshold"] = round(float(rng.uniform(
            diam_min + (diam_max - diam_min) * 0.6,
            diam_max + 2.0,
        )), 1)
        rules["big_spacing"] = round(default_spacing + float(rng.uniform(5, 12)), 1)

    if level >= 3:
        # Aggregate: total hole area limit
        rules["max_area_pct"] = round(float(rng.uniform(1.5, 5)), 1)

    if level >= 4:
        # Chained: if area exceeded AND hole is big, it's critical
        rules["critical_penalty"] = round(float(rng.uniform(2, 5)), 1)

    return rules


def rules_to_text(rules: dict[str, Any]) -> str:
    """Convert rules to spec text shown to the model."""
    lines = []
    lines.append(f"R1: All hole diameters must be between {rules['diam_min']} mm and {rules['diam_max']} mm.")
    lines.append(f"R2: Minimum center-to-center spacing between any two holes: {rules['default_spacing']} mm.")

    if "big_threshold" in rules:
        lines.append(
            f"R3: IF a hole's diameter exceeds {rules['big_threshold']} mm, "
            f"THEN its spacing to ALL other holes must be at least {rules['big_spacing']} mm "
            f"(overrides R2 for that hole)."
        )

    if "max_area_pct" in rules:
        lines.append(
            f"R4: The total hole area (sum of pi*r^2 for all holes) must not exceed "
            f"{rules['max_area_pct']}% of the plate area."
        )

    if "critical_penalty" in rules:
        lines.append(
            f"R5: IF R4 is violated AND a hole triggered R3, add a {rules['critical_penalty']} mm "
            f"critical penalty for each such hole."
        )

    return "\n".join(lines)


def evaluate_rules(
    holes: list[dict[str, Any]],
    rules: dict[str, Any],
    plate_w_mm: float,
    plate_h_mm: float,
    ppm: float,
) -> dict[str, Any]:
    """Evaluate all rules, return reasoning chain + total violation."""
    level = rules["level"]
    measurement_lines: list[str] = []
    reasoning_lines: list[str] = []
    total_violation = 0.0

    # Measurements
    for h in holes:
        measurement_lines.append(f"{h['label']}: {h['d_mm']}mm")

    # R1: diameter check
    reasoning_lines.append(f"R1 (diameter {rules['diam_min']}-{rules['diam_max']}mm):")
    r1_violations: dict[str, float] = {}
    for h in holes:
        if h["d_mm"] < rules["diam_min"]:
            v = round(rules["diam_min"] - h["d_mm"], 2)
            r1_violations[h["label"]] = v
            total_violation += v
            reasoning_lines.append(f"  {h['label']}: {h['d_mm']} < {rules['diam_min']}. Violation: {v}mm.")
        elif h["d_mm"] > rules["diam_max"]:
            v = round(h["d_mm"] - rules["diam_max"], 2)
            r1_violations[h["label"]] = v
            total_violation += v
            reasoning_lines.append(f"  {h['label']}: {h['d_mm']} > {rules['diam_max']}. Violation: {v}mm.")
        else:
            reasoning_lines.append(f"  {h['label']}: {h['d_mm']} OK.")

    # R2: default spacing
    reasoning_lines.append(f"R2 (spacing >= {rules['default_spacing']}mm):")
    # Compute pairwise distances in mm
    distances: dict[str, float] = {}
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            dist_px = np.sqrt(
                (holes[i]["cx"] - holes[j]["cx"]) ** 2 +
                (holes[i]["cy"] - holes[j]["cy"]) ** 2
            )
            dist_mm = round(dist_px / ppm, 1)
            key = f"{holes[i]['label']}-{holes[j]['label']}"
            distances[key] = dist_mm

    # Identify big holes (for R3)
    big_holes: set[str] = set()
    if level >= 2 and "big_threshold" in rules:
        for h in holes:
            if h["d_mm"] > rules["big_threshold"]:
                big_holes.add(h["label"])

    r2_violations: dict[str, float] = {}
    for pair, dist in distances.items():
        h1_label, h2_label = pair.split("-")
        # Check if R3 overrides for this pair
        if level >= 2 and (h1_label in big_holes or h2_label in big_holes):
            # R3 handles this pair, skip R2
            reasoning_lines.append(f"  {pair}: {dist}mm (deferred to R3 - big hole involved).")
        else:
            if dist < rules["default_spacing"]:
                v = round(rules["default_spacing"] - dist, 2)
                r2_violations[pair] = v
                total_violation += v
                reasoning_lines.append(f"  {pair}: {dist}mm < {rules['default_spacing']}mm. Violation: {v}mm.")
            else:
                reasoning_lines.append(f"  {pair}: {dist}mm OK.")

    # R3: conditional spacing for big holes
    r3_triggered = False
    r3_violations: dict[str, float] = {}
    if level >= 2 and big_holes:
        r3_triggered = True
        reasoning_lines.append(
            f"R3 (holes > {rules['big_threshold']}mm need spacing >= {rules['big_spacing']}mm):"
        )
        reasoning_lines.append(f"  Big holes: {', '.join(sorted(big_holes))}")

        for pair, dist in distances.items():
            h1_label, h2_label = pair.split("-")
            if h1_label in big_holes or h2_label in big_holes:
                if dist < rules["big_spacing"]:
                    v = round(rules["big_spacing"] - dist, 2)
                    r3_violations[pair] = v
                    total_violation += v
                    reasoning_lines.append(
                        f"  {pair}: {dist}mm < {rules['big_spacing']}mm. Violation: {v}mm."
                    )
                else:
                    reasoning_lines.append(f"  {pair}: {dist}mm OK.")
    elif level >= 2:
        reasoning_lines.append(
            f"R3: No holes exceed {rules['big_threshold']}mm. Rule does not apply."
        )

    # R4: area check
    r4_violated = False
    if level >= 3 and "max_area_pct" in rules:
        plate_area = plate_w_mm * plate_h_mm
        total_area = sum(np.pi * (h["d_mm"] / 2) ** 2 for h in holes)
        area_pct = round(total_area / plate_area * 100, 1)
        max_pct = rules["max_area_pct"]

        reasoning_lines.append(f"R4 (total hole area <= {max_pct}% of plate):")
        reasoning_lines.append(f"  Plate area: {plate_w_mm} x {plate_h_mm} = {plate_area:.0f} mm²")
        reasoning_lines.append(f"  Total hole area: {total_area:.1f} mm² ({area_pct}%)")

        if area_pct > max_pct:
            r4_violated = True
            v = round((area_pct - max_pct) / 100 * plate_area / 100, 2)  # penalty in mm scale
            total_violation += v
            reasoning_lines.append(f"  {area_pct}% > {max_pct}%. Violation: {v}mm equivalent.")
        else:
            reasoning_lines.append(f"  {area_pct}% <= {max_pct}%. OK.")

    # R5: chained conditional
    if level >= 4 and "critical_penalty" in rules:
        reasoning_lines.append(f"R5 (R4 violated AND R3 triggered → critical penalty):")
        if r4_violated and r3_triggered and big_holes:
            for h_label in sorted(big_holes):
                penalty = rules["critical_penalty"]
                total_violation += penalty
                reasoning_lines.append(
                    f"  {h_label}: R4 violated + triggered R3. Critical penalty: {penalty}mm."
                )
        elif not r4_violated:
            reasoning_lines.append(f"  R4 not violated. Rule does not apply.")
        elif not r3_triggered:
            reasoning_lines.append(f"  R3 not triggered. Rule does not apply.")
        else:
            reasoning_lines.append(f"  No big holes. Rule does not apply.")

    total_violation = round(total_violation, 2)
    reasoning_lines.append(f"Total violation: {total_violation}mm")

    full_answer = "\n".join(measurement_lines) + "\n" + "\n".join(reasoning_lines)

    return {
        "total": total_violation,
        "answer": full_answer,
        "r1_violations": r1_violations,
        "r2_violations": r2_violations,
        "r3_triggered": r3_triggered,
        "r3_violations": r3_violations,
        "r4_violated": r4_violated,
        "big_holes": list(big_holes),
    }


# ========================
# GENERATOR
# ========================

def generate_dataset(n_per_level: int = 100, n_test_per_level: int = 50, seed: int = 90) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpl_patches

    rng = np.random.default_rng(seed)
    out = Path(DATASET_DIR)

    for split in ["train", "test"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    for split in ["train", "test"]:
        n_per = n_per_level if split == "train" else n_test_per_level
        samples: list[dict[str, Any]] = []

        for level in [1, 2, 3, 4]:
            generated = 0
            attempts = 0

            while generated < n_per and attempts < n_per * 5:
                attempts += 1
                n_holes = int(rng.integers(3, 6))
                rules = make_rules(rng, level)

                # Generate diameters: mix of in-spec and out-of-spec
                diameters: list[float] = []
                for _ in range(n_holes):
                    if rng.random() < 0.5:
                        # In spec
                        d = round(float(rng.uniform(rules["diam_min"] + 0.3, rules["diam_max"] - 0.3)), 2)
                    else:
                        # Might be out of spec
                        d = round(float(rng.uniform(3, 28)), 2)
                    diameters.append(d)

                # Sometimes force all in-spec for zero-violation samples
                if rng.random() < 0.4:
                    diameters = [round(float(rng.uniform(rules["diam_min"] + 0.5, rules["diam_max"] - 0.5)), 2)
                                 for _ in range(n_holes)]

                sb_mm = int(rng.choice(SCALE_BAR_VALUES))
                canvas_w = int(rng.integers(600, 900))
                canvas_h = int(rng.integers(450, 700))

                max_diam = max(diameters)
                min_diam = min(diameters)
                ppm_min = max(12 / min_diam, 30 / sb_mm)
                ppm_max = min(0.3 * canvas_w / max_diam, 0.3 * canvas_w / sb_mm)
                if ppm_min >= ppm_max:
                    canvas_w = int(canvas_w * 1.5)
                    canvas_h = int(canvas_h * 1.5)
                    ppm_max = min(0.3 * canvas_w / max_diam, 0.3 * canvas_w / sb_mm)
                if ppm_min >= ppm_max:
                    continue

                ppm = float(rng.uniform(ppm_min, min(ppm_max, ppm_min * 5)))
                sb_px = sb_mm * ppm

                plate_margin = 50
                plate_w_px = canvas_w - 2 * plate_margin
                plate_h_px = canvas_h - 2 * plate_margin - 40
                plate_x = float(plate_margin)
                plate_y = 70.0
                plate_w_mm = round(plate_w_px / ppm, 1)
                plate_h_mm = round(plate_h_px / ppm, 1)

                # Place holes
                holes: list[dict[str, Any]] = []
                ok = True
                for i, d_mm in enumerate(diameters):
                    d_px = d_mm * ppm
                    placed = False
                    for _ in range(100):
                        cx_lo = plate_x + d_px / 2 + 15
                        cx_hi = plate_x + plate_w_px - d_px / 2 - 15
                        cy_lo = plate_y + d_px / 2 + 15
                        cy_hi = plate_y + plate_h_px - d_px / 2 - 15
                        if cx_lo >= cx_hi or cy_lo >= cy_hi:
                            break
                        cx = float(rng.uniform(cx_lo, cx_hi))
                        cy = float(rng.uniform(cy_lo, cy_hi))
                        if all(np.sqrt((cx - h["cx"]) ** 2 + (cy - h["cy"]) ** 2) >
                               (d_px + h["d_px"]) / 2 + 8 for h in holes):
                            placed = True
                            break
                    if not placed:
                        ok = False
                        break
                    holes.append({
                        "label": f"H{i + 1}",
                        "d_mm": d_mm, "d_px": round(d_px, 1),
                        "cx": round(cx, 1), "cy": round(cy, 1),
                    })

                if not ok:
                    continue

                # Evaluate rules
                result = evaluate_rules(holes, rules, plate_w_mm, plate_h_mm, ppm)

                idx = len(samples)
                sample = {
                    "idx": idx,
                    "level": level,
                    "n_holes": n_holes,
                    "holes": holes,
                    "diameters_mm": diameters,
                    "rules": rules,
                    "rules_text": rules_to_text(rules),
                    "gt_total": result["total"],
                    "gt_answer": result["answer"],
                    "r3_triggered": result["r3_triggered"],
                    "r4_violated": result.get("r4_violated", False),
                    "big_holes": result.get("big_holes", []),
                    "plate_w_mm": plate_w_mm,
                    "plate_h_mm": plate_h_mm,
                    "scale_bar_mm": sb_mm,
                    "sb_px": round(sb_px, 1),
                    "ppm": round(ppm, 4),
                    "canvas_w": canvas_w,
                    "canvas_h": canvas_h,
                    "plate_x": plate_x,
                    "plate_y": plate_y,
                    "plate_w_px": plate_w_px,
                    "plate_h_px": plate_h_px,
                }
                samples.append(sample)

                # Render
                cw, ch = canvas_w, canvas_h
                fig, ax = plt.subplots(1, 1, figsize=(cw / 150, ch / 150), dpi=150)
                ax.set_xlim(0, cw)
                ax.set_ylim(0, ch)
                ax.set_aspect("equal")
                ax.axis("off")

                rect = mpl_patches.Rectangle(
                    (plate_x, plate_y), plate_w_px, plate_h_px,
                    linewidth=1.5, edgecolor="black", facecolor="#F0F0EA", zorder=1)
                ax.add_patch(rect)

                for h in holes:
                    circle = mpl_patches.Circle(
                        (h["cx"], h["cy"]), h["d_px"] / 2,
                        linewidth=1.0, edgecolor="black", facecolor="#333333", zorder=2)
                    ax.add_patch(circle)
                    ax.text(h["cx"], h["cy"] + h["d_px"] / 2 + 10, h["label"],
                            ha="center", va="bottom", fontsize=6, fontweight="bold", zorder=3)

                sb_x = plate_x
                sb_y = plate_y - 35
                ax.plot([sb_x, sb_x + sb_px], [sb_y, sb_y], color="black", linewidth=2, zorder=5)
                ax.plot([sb_x, sb_x], [sb_y - 5, sb_y + 5], color="black", linewidth=1.5, zorder=5)
                ax.plot([sb_x + sb_px, sb_x + sb_px], [sb_y - 5, sb_y + 5], color="black", linewidth=1.5, zorder=5)
                ax.text(sb_x + sb_px / 2, sb_y - 10, f"{sb_mm} mm",
                        ha="center", va="top", fontsize=7, zorder=5)

                fig.savefig(str(out / split / f"image_{idx:04d}.png"),
                            dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
                plt.close(fig)
                generated += 1

            print(f"  {split} level {level}: {generated}/{n_per}")

        with open(out / split / "metadata.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        # Shortcut verification per level
        for level in [1, 2, 3, 4]:
            level_samples = [s for s in samples if s["level"] == level]
            if not level_samples:
                continue
            totals = [s["gt_total"] for s in level_samples]
            n_zeros = sum(1 for t in totals if t == 0)
            n_r3 = sum(1 for s in level_samples if s.get("r3_triggered", False))
            n_r4 = sum(1 for s in level_samples if s.get("r4_violated", False))

            # Shortcut check: spec tightness vs total
            tightness = [s["rules"]["diam_max"] - s["rules"]["diam_min"] for s in level_samples]
            corr_tight = abs(np.corrcoef(tightness, totals)[0, 1]) if len(set(tightness)) > 1 else 0
            # Shortcut check: n_holes vs total
            n_holes_list = [s["n_holes"] for s in level_samples]
            corr_nholes = abs(np.corrcoef(n_holes_list, totals)[0, 1]) if len(set(n_holes_list)) > 1 else 0

            print(f"  L{level}: zeros={n_zeros}/{len(level_samples)}, "
                  f"r3_triggered={n_r3}, r4_violated={n_r4}, "
                  f"total_range={min(totals):.1f}-{max(totals):.1f}, "
                  f"corr(tight,total)={corr_tight:.2f}, corr(nholes,total)={corr_nholes:.2f}")

    print(f"\nDone. Output in {DATASET_DIR}/")


# ========================
# UTILS
# ========================

SYSTEM_PROMPT: str = (
    "You are inspecting a technical drawing for compliance with multiple rules.\n"
    "{rules}\n\n"
    "Use the scale bar to measure each hole diameter, the distances between holes, "
    "and the plate dimensions. "
    "First list each measurement. Then check each rule, noting which conditionals apply. "
    "Finally give the total violation amount.\n"
    "Format:\n"
    "[measurements]\n"
    "[rule-by-rule checking]\n"
    "Total violation: [number]mm"
)


def make_prompt(rules_text: str) -> str:
    return SYSTEM_PROMPT.format(rules=rules_text)


def parse_total(text: str) -> float | None:
    m = re.search(r'Total\s*violation:?\s*([\d.]+)', text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    m = re.search(r'Total:?\s*([\d.]+)', text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def load_samples(split: str) -> list[dict[str, Any]]:
    with open(Path(DATASET_DIR) / split / "metadata.jsonl") as f:
        return [json.loads(l) for l in f]


def infer_one(model, processor, image_path: str, rules_text: str,
              do_sample: bool = False, temperature: float = 0.7,
              max_tokens: int | None = None) -> str:
    image = Image.open(image_path).convert("RGB")
    prompt = make_prompt(rules_text)
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    gen_kwargs = {"max_new_tokens": max_tokens or MAX_NEW_TOKENS}
    if do_sample:
        gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
    else:
        gen_kwargs["do_sample"] = False
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ========================
# SFT (levels 1-2 only)
# ========================

def run_sft() -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output = "checkpoints_cond_sft"
    print("\n=== Conditional Rules SFT (levels 1-2) ===\n")

    all_samples = load_samples("train")
    samples = [s for s in all_samples if s["level"] <= 2]
    total_steps = len(samples) * NUM_EPOCHS_SFT
    print(f"Train: {len(samples)} (levels 1-2), steps: {total_steps}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                              target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    os.makedirs(output, exist_ok=True)
    running_loss: list[float] = []
    global_step = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS_SFT):
        rng = np.random.default_rng(42 + epoch)
        for idx in rng.permutation(len(samples)):
            global_step += 1
            s = samples[idx]
            path = str(Path(DATASET_DIR) / "train" / f"image_{s['idx']:04d}.png")
            image = Image.open(path).convert("RGB")

            prompt = make_prompt(s["rules_text"])
            answer = s["gt_answer"]

            # Answer-only loss masking
            user_msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            prompt_text = processor.apply_chat_template(
                user_msgs, add_generation_prompt=True, tokenize=False)
            prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]

            full_msgs = [
                {"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": answer},
            ]
            full_text = processor.apply_chat_template(full_msgs, tokenize=False)
            inputs = processor(text=[full_text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            labels = inputs["input_ids"].clone()
            labels[0, :prompt_len] = -100

            model.train()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            if loss is None or torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss.append(loss.item())

            if global_step % LOG_EVERY == 0:
                avg = np.mean(running_loss[-LOG_EVERY:])
                elapsed = time.time() - start_time
                spm = global_step / (elapsed / 60) if elapsed > 0 else 1
                eta = (total_steps - global_step) / spm if spm > 0 else 0
                print(f"  Step {global_step}/{total_steps} | loss={avg:.4f} | ETA={eta:.0f}m")

    final = os.path.join(output, "final")
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"\n✓ SFT complete.")


# ========================
# EVAL
# ========================

def run_eval(method: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt = f"checkpoints_cond_{method}/final"
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}")
        return

    print(f"\n=== Conditional Rules {method} Eval ===\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()
    if method == "grpo":
        sft_path = "checkpoints_cond_sft/final"
        if os.path.exists(sft_path):
            model = PeftModel.from_pretrained(model, sft_path)
            model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, ckpt)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    samples = load_samples("test")
    tolerance = 1.0

    by_level: dict[int, dict[str, Any]] = {}
    for i, s in enumerate(samples):
        lv = s["level"]
        if lv not in by_level:
            by_level[lv] = {"correct": 0, "parsed": 0, "total": 0, "maes": []}

        path = str(Path(DATASET_DIR) / "test" / f"image_{s['idx']:04d}.png")
        raw = infer_one(model, processor, path, s["rules_text"])
        pred = parse_total(raw)
        by_level[lv]["total"] += 1
        if pred is not None:
            by_level[lv]["parsed"] += 1
            error = abs(pred - s["gt_total"])
            by_level[lv]["maes"].append(error)
            if error <= tolerance:
                by_level[lv]["correct"] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(samples)}]")

    print(f"\n{method} results by level:")
    print(f"{'Level':<7} {'Correct':>8} {'Parsed':>8} {'MAE':>8}")
    print("-" * 35)

    results = {}
    for lv in sorted(by_level.keys()):
        d = by_level[lv]
        mae = np.mean(d["maes"]) if d["maes"] else float("inf")
        acc = d["correct"] / max(d["parsed"], 1) * 100
        print(f"  L{lv:<5} {d['correct']:>5}/{d['parsed']:<3} {d['parsed']:>5}/{d['total']:<3} {mae:>7.2f}")
        results[f"L{lv}"] = {
            "correct": d["correct"], "parsed": d["parsed"],
            "total": d["total"], "mae": round(mae, 2), "accuracy": round(acc, 1),
        }

    os.makedirs("results_conditional", exist_ok=True)
    with open(f"results_conditional/{method}.json", "w") as f:
        json.dump(results, f, indent=2)

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ========================
# NO-IMAGE ABLATION
# ========================

def run_no_image_ablation(method: str) -> None:
    """Run eval with blank white images to detect text shortcuts."""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt = f"checkpoints_cond_{method}/final"
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}")
        return

    print(f"\n=== No-Image Ablation ({method}) ===\n")
    print("Using blank white images. If accuracy > chance, text shortcuts exist.\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()
    if method == "grpo":
        sft_path = "checkpoints_cond_sft/final"
        if os.path.exists(sft_path):
            model = PeftModel.from_pretrained(model, sft_path)
            model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, ckpt)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    # Create a blank white image
    blank_path = os.path.join(DATASET_DIR, "blank.png")
    blank = Image.new("RGB", (700, 550), "white")
    blank.save(blank_path)

    samples = load_samples("test")
    tolerance = 1.0

    by_level: dict[int, dict[str, Any]] = {}
    for i, s in enumerate(samples):
        lv = s["level"]
        if lv not in by_level:
            by_level[lv] = {"correct": 0, "parsed": 0, "total": 0, "maes": []}

        # Use blank image instead of real one
        raw = infer_one(model, processor, blank_path, s["rules_text"])
        pred = parse_total(raw)
        by_level[lv]["total"] += 1
        if pred is not None:
            by_level[lv]["parsed"] += 1
            error = abs(pred - s["gt_total"])
            by_level[lv]["maes"].append(error)
            if error <= tolerance:
                by_level[lv]["correct"] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(samples)}]")

    print(f"\nNo-image ablation ({method}):")
    print(f"{'Level':<7} {'Correct':>8} {'Parsed':>8} {'MAE':>8}")
    print("-" * 35)

    for lv in sorted(by_level.keys()):
        d = by_level[lv]
        mae = np.mean(d["maes"]) if d["maes"] else float("inf")
        print(f"  L{lv:<5} {d['correct']:>5}/{d['parsed']:<3} {d['parsed']:>5}/{d['total']:<3} {mae:>7.2f}")

    # Compare to mean-guess baseline
    all_totals = [s["gt_total"] for s in samples]
    mean_total = np.mean(all_totals)
    mean_guess_mae = np.mean([abs(t - mean_total) for t in all_totals])
    print(f"\n  Mean-guess MAE (always predict {mean_total:.1f}): {mean_guess_mae:.2f}mm")
    print(f"  If no-image MAE ≈ mean-guess MAE, no text shortcut exists.")

    os.makedirs("results_conditional", exist_ok=True)
    with open(f"results_conditional/no_image_{method}.json", "w") as f:
        results = {}
        for lv in sorted(by_level.keys()):
            d = by_level[lv]
            mae = np.mean(d["maes"]) if d["maes"] else float("inf")
            results[f"L{lv}"] = {"correct": d["correct"], "parsed": d["parsed"],
                                  "total": d["total"], "mae": round(mae, 2)}
        json.dump(results, f, indent=2)

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ========================
# BINNING
# ========================

def run_binning() -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    sft_path = "checkpoints_cond_sft/final"
    if not os.path.exists(sft_path):
        print("ERROR: Run --sft first.")
        return

    print("\n=== Difficulty Binning (pass@8, all levels) ===\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, sft_path)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    all_samples = load_samples("train")
    samples = [s for s in all_samples if s["level"] >= 3]
    tolerance = 1.0
    BIN_MAX_TOKENS = 256  # shorter for speed, just need the total

    bins: dict[str, list[int]] = {"easy": [], "medium": [], "hard": []}
    by_level_bins: dict[int, dict[str, int]] = {}

    for i, s in enumerate(samples):
        path = str(Path(DATASET_DIR) / "train" / f"image_{s['idx']:04d}.png")
        correct = 0
        for _ in range(NUM_GENERATIONS):
            raw = infer_one(model, processor, path, s["rules_text"],
                           do_sample=True, temperature=0.7,
                           max_tokens=BIN_MAX_TOKENS)
            pred = parse_total(raw)
            if pred is not None and abs(pred - s["gt_total"]) <= tolerance:
                correct += 1

        rate = correct / NUM_GENERATIONS
        if rate >= EASY_THRESHOLD:
            bins["easy"].append(s["idx"])
            cat = "easy"
        elif rate >= MEDIUM_LOW:
            bins["medium"].append(s["idx"])
            cat = "medium"
        else:
            bins["hard"].append(s["idx"])
            cat = "hard"

        lv = s["level"]
        if lv not in by_level_bins:
            by_level_bins[lv] = {"easy": 0, "medium": 0, "hard": 0}
        by_level_bins[lv][cat] += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(samples)}] "
                  f"easy={len(bins['easy'])} medium={len(bins['medium'])} hard={len(bins['hard'])}")

    os.makedirs("results_conditional", exist_ok=True)
    with open("results_conditional/bins.json", "w") as f:
        json.dump(bins, f, indent=2)

    print(f"\nBinning complete:")
    print(f"  Easy: {len(bins['easy'])}, Medium: {len(bins['medium'])}, Hard: {len(bins['hard'])}")
    print(f"\n  By level:")
    for lv in sorted(by_level_bins.keys()):
        d = by_level_bins[lv]
        print(f"    L{lv}: easy={d['easy']} medium={d['medium']} hard={d['hard']}")

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ========================
# GRPO on medium bin
# ========================

def sapo_gate(ratio: torch.Tensor, advantage: float) -> torch.Tensor:
    tau = TAU_POS if advantage > 0 else TAU_NEG
    return torch.sigmoid(tau * (ratio - 1.0))


def run_grpo() -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    sft_path = "checkpoints_cond_sft/final"
    output = "checkpoints_cond_grpo"
    bins_path = "results_conditional/bins.json"

    if not os.path.exists(sft_path):
        print("ERROR: Run --sft first.")
        return
    if not os.path.exists(bins_path):
        print("ERROR: Run --bin first.")
        return

    with open(bins_path) as f:
        bins = json.load(f)

    medium_ids = set(bins["medium"])
    all_samples = load_samples("train")
    samples = [s for s in all_samples if s["idx"] in medium_ids]

    print(f"\n=== GRPO on medium bin ({len(samples)} samples) ===\n")

    if len(samples) < 10:
        print("Too few medium samples. Adjust thresholds or dataset.")
        return

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, sft_path)
    model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    ref_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    if os.path.exists(WARMSTART_CKPT):
        ref_model = PeftModel.from_pretrained(ref_model, WARMSTART_CKPT)
        ref_model = ref_model.merge_and_unload()
    ref_model = PeftModel.from_pretrained(ref_model, sft_path)
    ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA,
                              target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    os.makedirs(output, exist_ok=True)
    log_path = os.path.join(output, "log.jsonl")
    tolerance = 1.0

    running_reward: list[float] = []
    running_loss: list[float] = []
    global_step = 0
    start_time = time.time()

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(samples))

    for si in indices:
        global_step += 1
        s = samples[si]
        gt = s["gt_total"]
        path = str(Path(DATASET_DIR) / "train" / f"image_{s['idx']:04d}.png")

        image = Image.open(path).convert("RGB")
        prompt = make_prompt(s["rules_text"])
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        completions: list[str] = []
        gen_ids_list: list[torch.Tensor] = []
        model.eval()
        for _ in range(NUM_GENERATIONS):
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=0.7, top_p=0.9)
            gids = out[0, input_len:].clone()
            gen_ids_list.append(gids)
            completions.append(processor.decode(gids, skip_special_tokens=True).strip())
            del out
        torch.cuda.empty_cache()

        rewards: list[float] = []
        for c in completions:
            pred = parse_total(c)
            if pred is None:
                rewards.append(-1.0)
            elif abs(pred - gt) <= tolerance:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        running_reward.extend(rewards)

        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in rewards]

        if all(abs(a) < 0.01 for a in advantages):
            del inputs, gen_ids_list
            torch.cuda.empty_cache()
            continue

        model.train()
        optimizer.zero_grad()
        total_loss = 0.0

        for gids, adv in zip(gen_ids_list, advantages):
            if len(gids) > MAX_NEW_TOKENS:
                gids = gids[:MAX_NEW_TOKENS]

            full_ids = torch.cat([inputs["input_ids"][0], gids]).unsqueeze(0)
            try:
                out_m = model(input_ids=full_ids,
                              pixel_values=inputs.get("pixel_values"),
                              image_grid_thw=inputs.get("image_grid_thw"))
            except RuntimeError:
                torch.cuda.empty_cache()
                continue

            logits = out_m.logits[0, input_len - 1:-1, :]
            lp = torch.log_softmax(logits, dim=-1)
            tlp = lp[range(len(gids)), gids]

            with torch.no_grad():
                ref_out = ref_model(input_ids=full_ids,
                                    pixel_values=inputs.get("pixel_values"),
                                    image_grid_thw=inputs.get("image_grid_thw"))
                ref_lp = torch.log_softmax(ref_out.logits[0, input_len - 1:-1, :], dim=-1)
                ref_tlp = ref_lp[range(len(gids)), gids]
                ratio = torch.exp(tlp - ref_tlp)

            gate = sapo_gate(ratio.detach(), adv)
            log_ratio = tlp - ref_tlp
            policy_loss = -(adv * gate * tlp).sum() / NUM_GENERATIONS
            kl_loss = KL_BETA * log_ratio.sum() / NUM_GENERATIONS
            loss = policy_loss + kl_loss
            loss.backward()
            total_loss += loss.item()

            del out_m, ref_out, logits, lp, ref_lp, tlp, ref_tlp, ratio, gate, loss, full_ids
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss.append(total_loss)

        del inputs, gen_ids_list
        torch.cuda.empty_cache()
        gc.collect()

        if global_step % LOG_EVERY == 0:
            avg_r = np.mean(running_reward[-LOG_EVERY * NUM_GENERATIONS:])
            n_correct = sum(1 for r in running_reward[-LOG_EVERY * NUM_GENERATIONS:] if r == 1.0)
            n_recent = min(LOG_EVERY * NUM_GENERATIONS, len(running_reward))
            elapsed = time.time() - start_time
            spm = global_step / (elapsed / 60) if elapsed > 0 else 1
            eta = (len(samples) - global_step) / spm if spm > 0 else 0

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": global_step, "reward": round(avg_r, 4),
                    "correct": n_correct, "n": n_recent,
                    "level": s["level"],
                }) + "\n")

            print(f"  Step {global_step}/{len(samples)} L{s['level']} | "
                  f"reward={avg_r:.3f} | correct={n_correct}/{n_recent} | ETA={eta:.0f}m")

    final = os.path.join(output, "final")
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"\n✓ GRPO complete.")


# ========================
# SUMMARY
# ========================

def run_summary() -> None:
    print(f"\n{'=' * 65}")
    print(f"  Conditional Rules: SFT vs GRPO by Level")
    print(f"{'=' * 65}\n")

    for method in ["sft", "grpo"]:
        path = f"results_conditional/{method}.json"
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            print(f"  {method}:")
            for lv in sorted(r.keys()):
                d = r[lv]
                print(f"    {lv}: {d['correct']}/{d['parsed']} correct "
                      f"({d['accuracy']}%), MAE={d['mae']}mm")
            print()

    bins_path = "results_conditional/bins.json"
    if os.path.exists(bins_path):
        with open(bins_path) as f:
            bins = json.load(f)
        print(f"  Bins: easy={len(bins['easy'])} medium={len(bins['medium'])} hard={len(bins['hard'])}")


def run_verify() -> None:
    """Run the ACTUAL generator logic without saving images, check rule stats."""
    rng = np.random.default_rng(90)
    
    print("\n=== Verify: Rule Triggering Statistics (using real generator logic) ===\n")
    
    for level in [1, 2, 3, 4]:
        n_target = 500
        stats = {
            "r1_violations": 0, "r2_violations": 0,
            "r3_triggered": 0, "r3_violations": 0,
            "r4_violated": 0, "r5_triggered": 0,
            "zero_total": 0, "total_sum": 0.0,
            "plate_areas": [], "hole_area_pcts": [],
        }
        generated = 0
        attempts = 0
        
        while generated < n_target and attempts < n_target * 5:
            attempts += 1
            n_holes = int(rng.integers(3, 6))
            rules = make_rules(rng, level)
            
            # === EXACT COPY of generator diameter logic ===
            diameters: list[float] = []
            for _ in range(n_holes):
                if rng.random() < 0.5:
                    d = round(float(rng.uniform(rules["diam_min"] + 0.3, rules["diam_max"] - 0.3)), 2)
                else:
                    d = round(float(rng.uniform(3, 28)), 2)
                diameters.append(d)
            if rng.random() < 0.4:
                diameters = [round(float(rng.uniform(rules["diam_min"] + 0.5, rules["diam_max"] - 0.5)), 2)
                             for _ in range(n_holes)]
            
            # === EXACT COPY of generator canvas/plate logic ===
            sb_mm = int(rng.choice(SCALE_BAR_VALUES))
            canvas_w = int(rng.integers(600, 900))
            canvas_h = int(rng.integers(450, 700))
            
            max_diam = max(diameters)
            min_diam = min(diameters)
            ppm_min = max(12 / min_diam, 30 / sb_mm)
            ppm_max = min(0.3 * canvas_w / max_diam, 0.3 * canvas_w / sb_mm)
            if ppm_min >= ppm_max:
                canvas_w = int(canvas_w * 1.5)
                canvas_h = int(canvas_h * 1.5)
                ppm_max = min(0.3 * canvas_w / max_diam, 0.3 * canvas_w / sb_mm)
            if ppm_min >= ppm_max:
                continue
            
            ppm = float(rng.uniform(ppm_min, min(ppm_max, ppm_min * 5)))
            
            plate_margin = 50
            plate_w_px = canvas_w - 2 * plate_margin
            plate_h_px = canvas_h - 2 * plate_margin - 40
            plate_x = float(plate_margin)
            plate_y = 70.0
            plate_w_mm = round(plate_w_px / ppm, 1)
            plate_h_mm = round(plate_h_px / ppm, 1)
            
            # === EXACT COPY of generator hole placement logic ===
            holes = []
            ok = True
            for i, d_mm in enumerate(diameters):
                d_px = d_mm * ppm
                placed = False
                for _ in range(100):
                    cx_lo = plate_x + d_px / 2 + 15
                    cx_hi = plate_x + plate_w_px - d_px / 2 - 15
                    cy_lo = plate_y + d_px / 2 + 15
                    cy_hi = plate_y + plate_h_px - d_px / 2 - 15
                    if cx_lo >= cx_hi or cy_lo >= cy_hi:
                        break
                    cx = float(rng.uniform(cx_lo, cx_hi))
                    cy = float(rng.uniform(cy_lo, cy_hi))
                    if all(np.sqrt((cx - h["cx"]) ** 2 + (cy - h["cy"]) ** 2) >
                           (d_px + h["d_px"]) / 2 + 8 for h in holes):
                        placed = True
                        break
                if not placed:
                    ok = False
                    break
                holes.append({
                    "label": f"H{i+1}", "d_mm": d_mm, "d_px": round(d_px, 1),
                    "cx": round(cx, 1), "cy": round(cy, 1),
                })
            
            if not ok:
                continue
            
            generated += 1
            
            # Track plate area stats
            plate_area = plate_w_mm * plate_h_mm
            total_hole_area = sum(np.pi * (h["d_mm"] / 2) ** 2 for h in holes)
            area_pct = total_hole_area / plate_area * 100
            stats["plate_areas"].append(plate_area)
            stats["hole_area_pcts"].append(area_pct)
            
            result = evaluate_rules(holes, rules, plate_w_mm, plate_h_mm, ppm)
            
            if result["r1_violations"]:
                stats["r1_violations"] += 1
            if result["r2_violations"]:
                stats["r2_violations"] += 1
            if result["r3_triggered"]:
                stats["r3_triggered"] += 1
            if result["r3_violations"]:
                stats["r3_violations"] += 1
            if result.get("r4_violated", False):
                stats["r4_violated"] += 1
            if result.get("r4_violated", False) and result["r3_triggered"] and result.get("big_holes"):
                stats["r5_triggered"] += 1
            if result["total"] == 0:
                stats["zero_total"] += 1
            stats["total_sum"] += result["total"]
        
        areas = stats["plate_areas"]
        pcts = stats["hole_area_pcts"]
        print(f"  Level {level} (N={generated}, attempts={attempts}):")
        print(f"    Plate area: {np.mean(areas):.0f}mm² (range {np.min(areas):.0f}-{np.max(areas):.0f})")
        print(f"    Hole area %: {np.mean(pcts):.1f}% (range {np.min(pcts):.1f}-{np.max(pcts):.1f}%)")
        print(f"    R1 violations: {stats['r1_violations']}/{generated} ({stats['r1_violations']/generated*100:.0f}%)")
        print(f"    R2 violations: {stats['r2_violations']}/{generated} ({stats['r2_violations']/generated*100:.0f}%)")
        if level >= 2:
            print(f"    R3 triggered:  {stats['r3_triggered']}/{generated} ({stats['r3_triggered']/generated*100:.0f}%)")
            print(f"    R3 violations: {stats['r3_violations']}/{generated} ({stats['r3_violations']/generated*100:.0f}%)")
        if level >= 3:
            print(f"    R4 violated:   {stats['r4_violated']}/{generated} ({stats['r4_violated']/generated*100:.0f}%)")
            if rules.get("max_area_pct"):
                print(f"    R4 threshold:  {rules['max_area_pct']}% (last sample)")
        if level >= 4:
            print(f"    R5 triggered:  {stats['r5_triggered']}/{generated} ({stats['r5_triggered']/generated*100:.0f}%)")
        print(f"    Zero total:    {stats['zero_total']}/{generated} ({stats['zero_total']/generated*100:.0f}%)")
        print(f"    Mean total:    {stats['total_sum']/generated:.1f}mm")
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--bin", action="store_true")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--eval", type=str, choices=["sft", "grpo"])
    parser.add_argument("--no-image", type=str, choices=["sft", "grpo"], dest="no_image")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.generate:
        generate_dataset()
    if args.verify:
        run_verify()
    if args.sft:
        run_sft()
    if args.bin:
        run_binning()
    if args.grpo:
        run_grpo()
    if args.eval:
        run_eval(args.eval)
    if args.no_image:
        run_no_image_ablation(args.no_image)
    if args.summary:
        run_summary()

    if not any([args.generate, args.verify, args.sft, args.bin, args.grpo, args.eval, args.no_image, args.summary]):
        parser.print_help()


if __name__ == "__main__":
    main()
