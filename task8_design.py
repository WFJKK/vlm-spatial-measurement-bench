"""Task 8: Design a compliant plate configuration.

Given: empty plate image with scale bar + text spec with constraints.
Output: hole positions and diameters that satisfy ALL constraints.

There are thousands of valid configurations. SFT can only demonstrate one.
RL explores the space and learns to generate valid layouts.

Reward:
  +1.0 if all constraints satisfied
  +0.5 * (fraction of constraints satisfied) otherwise
  -0.5 if output is unparseable
  -1.0 if holes overlap or are outside plate

Usage:
    python3 task8_design.py --generate
    python3 task8_design.py --baseline
    python3 task8_design.py --sft
    python3 task8_design.py --eval sft
    python3 task8_design.py --grpo
    python3 task8_design.py --eval grpo
    python3 task8_design.py --summary
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
import torch
from PIL import Image

MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"
WARMSTART_CKPT: str = "checkpoints_task1_sft3/final"
DATASET_DIR: str = "dataset_task8"
LORA_RANK: int = 64
LORA_ALPHA: int = 128
LEARNING_RATE: float = 5e-6
NUM_EPOCHS: int = 3
LOG_EVERY: int = 10

# GRPO
NUM_GENERATIONS: int = 4
MAX_NEW_TOKENS: int = 256
KL_BETA: float = 0.2
TAU_POS: float = 1.0
TAU_NEG: float = 1.05

SCALE_BAR_VALUES: list[int] = [10, 15, 20, 25]


# ========================
# GENERATOR
# ========================

def generate_spec(rng: np.random.Generator) -> dict[str, Any]:
    """Generate a random but feasible spec."""
    n_holes = int(rng.integers(3, 6))
    diam_min = round(float(rng.uniform(8, 12)), 1)
    diam_max = round(diam_min + float(rng.uniform(3, 6)), 1)
    min_spacing = round(float(rng.uniform(15, 25)), 1)
    min_edge = round(float(rng.uniform(6, 12)), 1)

    return {
        "n_holes": n_holes,
        "diam_min": diam_min,
        "diam_max": diam_max,
        "min_spacing": min_spacing,
        "min_edge": min_edge,
    }


def spec_to_text(spec: dict[str, Any]) -> str:
    return (
        f"Design a plate with exactly {spec['n_holes']} holes. "
        f"All hole diameters must be between {spec['diam_min']} mm and {spec['diam_max']} mm. "
        f"Minimum center-to-center spacing between any two holes must be {spec['min_spacing']} mm. "
        f"Minimum distance from any hole center to any plate edge must be {spec['min_edge']} mm."
    )


def generate_one_valid_layout(
    rng: np.random.Generator, spec: dict[str, Any],
    plate_w_mm: float, plate_h_mm: float,
) -> list[dict[str, float]] | None:
    """Generate one valid layout. Returns None if can't find one in 200 attempts."""
    n = spec["n_holes"]
    for _ in range(200):
        holes = []
        ok = True
        for i in range(n):
            d = round(float(rng.uniform(spec["diam_min"], spec["diam_max"])), 1)
            for _ in range(50):
                x = round(float(rng.uniform(spec["min_edge"], plate_w_mm - spec["min_edge"])), 1)
                y = round(float(rng.uniform(spec["min_edge"], plate_h_mm - spec["min_edge"])), 1)
                valid = all(
                    np.sqrt((x - h["x"]) ** 2 + (y - h["y"]) ** 2) >= spec["min_spacing"]
                    for h in holes
                )
                if valid:
                    break
            else:
                ok = False
                break
            holes.append({"label": f"H{i+1}", "x": x, "y": y, "d": d})
        if ok:
            return holes
    return None


def generate_dataset(n_train: int = 500, n_test: int = 100, seed: int = 60) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpl_patches

    rng = np.random.default_rng(seed)
    out = Path(DATASET_DIR)

    for split, n in [("train", n_train), ("test", n_test)]:
        (out / split).mkdir(parents=True, exist_ok=True)
        samples: list[dict[str, Any]] = []
        skipped = 0

        for idx in range(n + 200):  # extra budget for failed layouts
            if len(samples) >= n:
                break

            spec = generate_spec(rng)
            sb_mm = int(rng.choice(SCALE_BAR_VALUES))

            # Plate dimensions in mm — large enough for the spec
            plate_w_mm = round(float(rng.uniform(80, 120)), 0)
            plate_h_mm = round(float(rng.uniform(60, 100)), 0)

            # Generate one valid layout as the SFT target
            layout = generate_one_valid_layout(rng, spec, plate_w_mm, plate_h_mm)
            if layout is None:
                skipped += 1
                continue

            # Rendering params
            canvas_w = int(rng.integers(600, 900))
            canvas_h = int(rng.integers(500, 750))
            plate_margin = 50

            ppm = min(
                (canvas_w - 2 * plate_margin) / plate_w_mm,
                (canvas_h - 2 * plate_margin - 40) / plate_h_mm,
            ) * 0.9
            sb_px = sb_mm * ppm

            render_plate_x = float(plate_margin)
            render_plate_y = 70.0
            render_plate_w = plate_w_mm * ppm
            render_plate_h = plate_h_mm * ppm

            sample = {
                "idx": len(samples),
                "spec": spec,
                "spec_text": spec_to_text(spec),
                "plate_w_mm": plate_w_mm,
                "plate_h_mm": plate_h_mm,
                "scale_bar_mm": sb_mm,
                "sb_px": round(sb_px, 1),
                "ppm": round(ppm, 4),
                "canvas_w": canvas_w,
                "canvas_h": canvas_h,
                "render_plate_x": render_plate_x,
                "render_plate_y": render_plate_y,
                "render_plate_w": round(render_plate_w, 1),
                "render_plate_h": round(render_plate_h, 1),
                "valid_layout": layout,  # one valid solution for SFT
            }
            samples.append(sample)

            # Render empty plate with scale bar and dimensions
            fig, ax = plt.subplots(1, 1, figsize=(canvas_w / 150, canvas_h / 150), dpi=150)
            ax.set_xlim(0, canvas_w)
            ax.set_ylim(0, canvas_h)
            ax.set_aspect("equal")
            ax.axis("off")

            rect = mpl_patches.Rectangle(
                (render_plate_x, render_plate_y), render_plate_w, render_plate_h,
                linewidth=1.5, edgecolor="black", facecolor="#F0F0EA", zorder=1,
            )
            ax.add_patch(rect)

            # Plate dimension labels
            ax.text(render_plate_x + render_plate_w / 2, render_plate_y + render_plate_h + 12,
                    f"{plate_w_mm} mm", ha="center", va="bottom", fontsize=7)
            ax.text(render_plate_x - 12, render_plate_y + render_plate_h / 2,
                    f"{plate_h_mm} mm", ha="right", va="center", fontsize=7, rotation=90)

            # Scale bar
            sb_x = render_plate_x
            sb_y = render_plate_y - 35
            ax.plot([sb_x, sb_x + sb_px], [sb_y, sb_y], color="black", linewidth=2, zorder=5)
            ax.plot([sb_x, sb_x], [sb_y - 5, sb_y + 5], color="black", linewidth=1.5, zorder=5)
            ax.plot([sb_x + sb_px, sb_x + sb_px], [sb_y - 5, sb_y + 5], color="black", linewidth=1.5, zorder=5)
            ax.text(sb_x + sb_px / 2, sb_y - 10, f"{sb_mm} mm",
                    ha="center", va="top", fontsize=7, zorder=5)

            img_path = str(out / split / f"image_{sample['idx']:04d}.png")
            fig.savefig(img_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
            plt.close(fig)

            if (len(samples)) % 100 == 0:
                print(f"  {split}: {len(samples)}/{n}")

        with open(out / split / "metadata.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        print(f"\n{split}: N={len(samples)}, skipped={skipped}")

    print(f"Done. Output in {DATASET_DIR}/")


# ========================
# VERIFICATION
# ========================

def parse_layout(text: str) -> list[dict[str, float]] | None:
    """Parse model output into hole positions and diameters."""
    holes = []
    # Match patterns like "H1: x=25.0, y=30.0, d=12.5" or "H1: 25.0, 30.0, 12.5"
    # Also match "H1: x=25, y=30, d=12.5"
    patterns = [
        r'H(\d+)\s*:\s*x\s*=\s*([\d.]+)\s*,?\s*y\s*=\s*([\d.]+)\s*,?\s*d\s*=\s*([\d.]+)',
        r'H(\d+)\s*:\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)',
        r'H(\d+)\s*[:\-]\s*x\s*[:=]\s*([\d.]+)\s*[,;]\s*y\s*[:=]\s*([\d.]+)\s*[,;]\s*d(?:iameter)?\s*[:=]\s*([\d.]+)',
    ]
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            for m in matches:
                holes.append({
                    "label": f"H{m[0]}",
                    "x": float(m[1]),
                    "y": float(m[2]),
                    "d": float(m[3]),
                })
            return holes if holes else None
    return None


def verify_layout(
    holes: list[dict[str, float]],
    spec: dict[str, Any],
    plate_w_mm: float,
    plate_h_mm: float,
) -> dict[str, Any]:
    """Check all constraints. Returns detailed results."""
    results: dict[str, Any] = {"constraints": [], "n_satisfied": 0, "n_total": 0, "valid": True}

    # Check hole count
    expected = spec["n_holes"]
    actual = len(holes)
    count_ok = actual == expected
    results["constraints"].append({"type": "hole_count", "ok": count_ok,
                                    "expected": expected, "actual": actual})
    results["n_total"] += 1
    if count_ok:
        results["n_satisfied"] += 1

    # Check each diameter
    for h in holes:
        ok = spec["diam_min"] <= h["d"] <= spec["diam_max"]
        results["constraints"].append({"type": "diameter", "hole": h["label"],
                                        "ok": ok, "value": h["d"]})
        results["n_total"] += 1
        if ok:
            results["n_satisfied"] += 1

    # Check edge distances
    for h in holes:
        left = h["x"]
        right = plate_w_mm - h["x"]
        top = h["y"]
        bottom = plate_h_mm - h["y"]
        min_dist = min(left, right, top, bottom)
        ok = min_dist >= spec["min_edge"]
        in_bounds = 0 <= h["x"] <= plate_w_mm and 0 <= h["y"] <= plate_h_mm
        results["constraints"].append({"type": "edge", "hole": h["label"],
                                        "ok": ok and in_bounds, "min_dist": round(min_dist, 1)})
        results["n_total"] += 1
        if ok and in_bounds:
            results["n_satisfied"] += 1
        if not in_bounds:
            results["valid"] = False

    # Check pairwise spacing
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            dist = np.sqrt((holes[i]["x"] - holes[j]["x"]) ** 2 +
                           (holes[i]["y"] - holes[j]["y"]) ** 2)
            ok = dist >= spec["min_spacing"]
            results["constraints"].append({
                "type": "spacing",
                "holes": f"{holes[i]['label']}-{holes[j]['label']}",
                "ok": ok, "dist": round(dist, 1),
            })
            results["n_total"] += 1
            if ok:
                results["n_satisfied"] += 1

    # Check overlaps
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            dist = np.sqrt((holes[i]["x"] - holes[j]["x"]) ** 2 +
                           (holes[i]["y"] - holes[j]["y"]) ** 2)
            if dist < (holes[i]["d"] + holes[j]["d"]) / 2:
                results["valid"] = False

    return results


def compute_reward(text: str, spec: dict[str, Any],
                   plate_w_mm: float, plate_h_mm: float) -> float:
    holes = parse_layout(text)
    if holes is None:
        return -0.5

    results = verify_layout(holes, spec, plate_w_mm, plate_h_mm)

    if not results["valid"]:
        return -1.0

    if results["n_total"] == 0:
        return -0.5

    fraction = results["n_satisfied"] / results["n_total"]

    if fraction == 1.0:
        return 1.0
    else:
        return -0.5 + fraction * 1.0  # ranges from -0.5 to +0.5


def layout_to_str(layout: list[dict[str, float]]) -> str:
    """Convert layout to model output format."""
    lines = []
    for h in layout:
        lines.append(f"{h['label']}: x={h['x']}, y={h['y']}, d={h['d']}")
    return "\n".join(lines)


def make_prompt(spec_text: str, plate_w_mm: float, plate_h_mm: float) -> str:
    return (
        f"You are designing a metal plate. The plate is {plate_w_mm} mm wide and {plate_h_mm} mm tall. "
        f"{spec_text} "
        f"Output each hole on a separate line as: H1: x=25.0, y=30.0, d=12.5 "
        f"where x,y are distances from the bottom-left corner in mm and d is the diameter in mm."
    )


# ========================
# BASELINE
# ========================

def run_baseline() -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("\n=== Task 8 Baseline ===\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    with open(Path(DATASET_DIR) / "test" / "metadata.jsonl") as f:
        samples = [json.loads(l) for l in f]

    rewards = []
    fully_compliant = 0
    parseable = 0

    for i, s in enumerate(samples):
        path = str(Path(DATASET_DIR) / "test" / f"image_{s['idx']:04d}.png")
        image = Image.open(path).convert("RGB")
        prompt = make_prompt(s["spec_text"], s["plate_w_mm"], s["plate_h_mm"])
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        raw = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        r = compute_reward(raw, s["spec"], s["plate_w_mm"], s["plate_h_mm"])
        rewards.append(r)

        if parse_layout(raw) is not None:
            parseable += 1
        if r == 1.0:
            fully_compliant += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/100] reward={np.mean(rewards):.3f} "
                  f"compliant={fully_compliant}/{i + 1} parsed={parseable}/{i + 1}")

    print(f"\nBaseline: reward={np.mean(rewards):.3f}, "
          f"fully_compliant={fully_compliant}/100, parseable={parseable}/100")

    os.makedirs("results_task8", exist_ok=True)
    with open("results_task8/baseline.json", "w") as f:
        json.dump({"reward": round(np.mean(rewards), 3),
                    "fully_compliant": fully_compliant, "parseable": parseable}, f, indent=2)

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ========================
# SFT
# ========================

def run_sft() -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output = "checkpoints_task8_sft"
    print("\n=== Task 8 SFT ===\n")

    with open(Path(DATASET_DIR) / "train" / "metadata.jsonl") as f:
        samples = [json.loads(l) for l in f]

    total_steps = len(samples) * NUM_EPOCHS
    print(f"Train: {len(samples)}, steps: {total_steps}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE,
    )

    os.makedirs(output, exist_ok=True)
    running_loss: list[float] = []
    global_step = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        rng = np.random.default_rng(42 + epoch)
        for idx in rng.permutation(len(samples)):
            global_step += 1
            s = samples[idx]
            path = str(Path(DATASET_DIR) / "train" / f"image_{s['idx']:04d}.png")
            image = Image.open(path).convert("RGB")

            prompt = make_prompt(s["spec_text"], s["plate_w_mm"], s["plate_h_mm"])
            answer = layout_to_str(s["valid_layout"])

            # Answer-only loss masking
            user_msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            prompt_text = processor.apply_chat_template(user_msgs, add_generation_prompt=True, tokenize=False)
            prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]

            full_msgs = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
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
    print(f"\n✓ Task 8 SFT complete.")


# ========================
# GRPO-SAPO
# ========================

def sapo_gate(ratio: torch.Tensor, advantage: float) -> torch.Tensor:
    tau = TAU_POS if advantage > 0 else TAU_NEG
    return torch.sigmoid(tau * (ratio - 1.0))


def run_grpo() -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    sft_path = "checkpoints_task8_sft/final"
    output = "checkpoints_task8_grpo"

    print("\n=== Task 8 GRPO-SAPO ===\n")

    if not os.path.exists(sft_path):
        print("ERROR: Run --sft first.")
        return

    with open(Path(DATASET_DIR) / "train" / "metadata.jsonl") as f:
        samples = [json.loads(l) for l in f]

    print(f"Train: {len(samples)}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, sft_path)
    model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    ref_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if os.path.exists(WARMSTART_CKPT):
        ref_model = PeftModel.from_pretrained(ref_model, WARMSTART_CKPT)
        ref_model = ref_model.merge_and_unload()
    ref_model = PeftModel.from_pretrained(ref_model, sft_path)
    ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE,
    )

    os.makedirs(output, exist_ok=True)
    log_path = os.path.join(output, "log.jsonl")

    running_reward: list[float] = []
    global_step = 0
    start_time = time.time()

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(samples))

    for idx in indices:
        global_step += 1
        s = samples[idx]
        path = str(Path(DATASET_DIR) / "train" / f"image_{s['idx']:04d}.png")

        image = Image.open(path).convert("RGB")
        prompt = make_prompt(s["spec_text"], s["plate_w_mm"], s["plate_h_mm"])
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
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
                    do_sample=True, temperature=0.8, top_p=0.9,
                )
            gids = out[0, input_len:].clone()
            gen_ids_list.append(gids)
            completions.append(processor.decode(gids, skip_special_tokens=True).strip())
            del out
        torch.cuda.empty_cache()

        rewards = [compute_reward(c, s["spec"], s["plate_w_mm"], s["plate_h_mm"])
                   for c in completions]
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
            out = model(
                input_ids=full_ids,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
            logits = out.logits[0, input_len - 1:-1, :]
            lp = torch.log_softmax(logits, dim=-1)
            tlp = lp[range(len(gids)), gids]

            with torch.no_grad():
                ref_out = ref_model(
                    input_ids=full_ids,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                )
                ref_lp = torch.log_softmax(ref_out.logits[0, input_len - 1:-1, :], dim=-1)
                ref_tlp = ref_lp[range(len(gids)), gids]

            with torch.no_grad():
                ratio = torch.exp(tlp - ref_tlp)
            gate = sapo_gate(ratio.detach(), adv)

            log_ratio = tlp - ref_tlp
            policy_loss = -(adv * gate * tlp).sum() / NUM_GENERATIONS
            kl_loss = KL_BETA * log_ratio.sum() / NUM_GENERATIONS
            loss = policy_loss + kl_loss
            loss.backward()
            total_loss += loss.item()

            del out, ref_out, logits, lp, ref_lp, tlp, ref_tlp, ratio, gate, loss, full_ids
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        del inputs, gen_ids_list
        torch.cuda.empty_cache()
        gc.collect()

        if global_step % LOG_EVERY == 0:
            avg_r = np.mean(running_reward[-LOG_EVERY * NUM_GENERATIONS:])
            elapsed = time.time() - start_time
            spm = global_step / (elapsed / 60) if elapsed > 0 else 1
            eta = (len(samples) - global_step) / spm if spm > 0 else 0

            n_compliant = sum(1 for r in running_reward[-LOG_EVERY * NUM_GENERATIONS:] if r == 1.0)
            n_recent = LOG_EVERY * NUM_GENERATIONS

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": global_step, "reward": round(avg_r, 4),
                    "compliant_rate": round(n_compliant / n_recent, 3),
                }) + "\n")

            preview = completions[0][:60].replace('\n', ' | ')
            print(f"  Step {global_step}/{len(samples)} | "
                  f"reward={avg_r:.3f} | compliant={n_compliant}/{n_recent} | "
                  f"ETA={eta:.0f}m | '{preview}'")

    final = os.path.join(output, "final")
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"\n✓ Task 8 GRPO-SAPO complete.")


# ========================
# EVAL
# ========================

def run_eval(method: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt = f"checkpoints_task8_{method}/final"
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}")
        return

    print(f"\n=== Task 8 {method} Eval ===\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()

    if method == "grpo":
        sft_path = "checkpoints_task8_sft/final"
        if os.path.exists(sft_path):
            model = PeftModel.from_pretrained(model, sft_path)
            model = model.merge_and_unload()

    model = PeftModel.from_pretrained(model, ckpt)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    with open(Path(DATASET_DIR) / "test" / "metadata.jsonl") as f:
        samples = [json.loads(l) for l in f]

    rewards = []
    fully_compliant = 0
    parseable = 0
    constraint_fractions: list[float] = []

    for i, s in enumerate(samples):
        path = str(Path(DATASET_DIR) / "test" / f"image_{s['idx']:04d}.png")
        image = Image.open(path).convert("RGB")
        prompt = make_prompt(s["spec_text"], s["plate_w_mm"], s["plate_h_mm"])
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        raw = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        r = compute_reward(raw, s["spec"], s["plate_w_mm"], s["plate_h_mm"])
        rewards.append(r)

        layout = parse_layout(raw)
        if layout is not None:
            parseable += 1
            v = verify_layout(layout, s["spec"], s["plate_w_mm"], s["plate_h_mm"])
            if v["n_total"] > 0:
                constraint_fractions.append(v["n_satisfied"] / v["n_total"])
        if r == 1.0:
            fully_compliant += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/100] compliant={fully_compliant}/{i + 1} "
                  f"parsed={parseable}/{i + 1}")

    result = {
        "method": method,
        "reward": round(np.mean(rewards), 3),
        "fully_compliant": fully_compliant,
        "parseable": parseable,
        "avg_constraint_fraction": round(np.mean(constraint_fractions), 3) if constraint_fractions else 0,
    }

    print(f"\n{method}: reward={result['reward']}, "
          f"fully_compliant={fully_compliant}/100, "
          f"parseable={parseable}/100, "
          f"avg_constraints_met={result['avg_constraint_fraction']}")

    os.makedirs("results_task8", exist_ok=True)
    with open(f"results_task8/{method}.json", "w") as f:
        json.dump(result, f, indent=2)

    del model
    torch.cuda.empty_cache()
    gc.collect()


def run_summary() -> None:
    print(f"\n{'=' * 65}")
    print(f"  Task 8: Design Compliant Plate (SFT vs GRPO-SAPO)")
    print(f"{'=' * 65}\n")

    print(f"{'Method':<12} {'Reward':>8} {'Compliant':>10} {'Parsed':>8} {'Constraints':>13}")
    print("-" * 55)

    for method in ["baseline", "sft", "grpo"]:
        path = f"results_task8/{method}.json"
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            print(f"{method:<12} {r['reward']:>8.3f} {r['fully_compliant']:>7}/100 "
                  f"{r['parseable']:>5}/100 {r.get('avg_constraint_fraction', 0):>12.3f}")

    print("-" * 55)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--eval", type=str, choices=["sft", "grpo"])
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.generate:
        generate_dataset()
    if args.baseline:
        run_baseline()
    if args.sft:
        run_sft()
    if args.grpo:
        run_grpo()
    if args.eval:
        run_eval(args.eval)
    if args.summary:
        run_summary()

    if not any([args.generate, args.baseline, args.sft, args.grpo, args.eval, args.summary]):
        parser.print_help()


if __name__ == "__main__":
    main()
