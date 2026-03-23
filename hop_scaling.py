"""Hop scaling experiment: at what complexity does SFT break?

Same visual elements (holes, plate, scale bar), increasingly complex questions.

2-hop: "Diameter difference between largest and smallest hole"
4-hop: "Total out-of-spec amount (all diameters vs spec range)"
6-hop: "Total non-compliance (diameters + spacing violations)"
8-hop: "Total non-compliance (diameters + spacing + edge distance)"

For each hop level: generate data, train SFT, evaluate.
Plot MAE vs hop count. When SFT stops improving, RL becomes relevant.

Usage:
    python3 hop_scaling.py --generate --hops 2
    python3 hop_scaling.py --generate --hops 4
    python3 hop_scaling.py --generate --hops 6
    python3 hop_scaling.py --generate --hops 8
    python3 hop_scaling.py --sft --hops 2
    python3 hop_scaling.py --eval --hops 2
    python3 hop_scaling.py --baseline --hops 2
    python3 hop_scaling.py --summary   # Compare all hop levels
"""

from __future__ import annotations

import argparse
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
LORA_RANK: int = 64
LORA_ALPHA: int = 128
LEARNING_RATE: float = 5e-6
NUM_EPOCHS: int = 3
LOG_EVERY: int = 10
SAVE_EVERY: int = 99999

SCALE_BAR_VALUES: list[int] = [10, 15, 20, 25, 30]
N_HOLES: int = 5


HOP_CONFIGS = {
    2: {
        "question": "What is the diameter difference between the largest and smallest hole in mm?",
        "system": "You are measuring holes in a technical drawing. Use the scale bar to measure all hole diameters. Respond with ONLY the difference in mm as a number, nothing else.",
    },
    4: {
        "question": "What is the total amount by which hole diameters are out of spec in mm? If all comply, answer 0.",
        "system": "You are inspecting a technical drawing. {spec} Use the scale bar to measure all hole diameters and sum up any violations. Respond with ONLY the total out-of-spec amount in mm as a number.",
    },
    6: {
        "question": "What is the total non-compliance in mm (diameter violations + spacing violations)? If all comply, answer 0.",
        "system": "You are inspecting a technical drawing. {spec} Use the scale bar to measure all hole diameters and distances between holes. Sum up all violations. Respond with ONLY the total in mm as a number.",
    },
    8: {
        "question": "What is the total non-compliance in mm (diameter + spacing + edge distance violations)? If all comply, answer 0.",
        "system": "You are inspecting a technical drawing. {spec} Use the scale bar to measure all hole diameters, distances between holes, and distances from holes to edges. Sum up all violations. Respond with ONLY the total in mm as a number.",
    },
}


def dataset_dir(hops: int) -> str:
    return f"dataset_hop{hops}"


def ckpt_dir(hops: int) -> str:
    return f"checkpoints_hop{hops}_sft"


# ========================
# GENERATOR
# ========================

def generate_sample(rng: np.random.Generator, idx: int, hops: int) -> dict[str, Any]:
    """Generate one sample for the given hop level."""
    import matplotlib
    matplotlib.use("Agg")

    # Fixed 5 holes for all levels
    diameters = [round(float(rng.uniform(5, 28)), 2) for _ in range(N_HOLES)]
    sb_mm = int(rng.choice(SCALE_BAR_VALUES))
    canvas_w = int(rng.integers(600, 1000))
    canvas_h = int(rng.integers(500, 800))

    max_diam = max(diameters)
    ppm_min = max(12 / min(diameters), 30 / sb_mm)
    ppm_max = min(0.3 * canvas_w / max_diam, 0.3 * canvas_w / sb_mm)
    if ppm_min >= ppm_max:
        canvas_w = int(canvas_w * 1.5)
        canvas_h = int(canvas_h * 1.5)
        ppm_max = min(0.3 * canvas_w / max_diam, 0.3 * canvas_w / sb_mm)
    if ppm_min >= ppm_max:
        ppm_max = ppm_min + 1.0

    ppm = float(rng.uniform(ppm_min, min(ppm_max, ppm_min * 3)))
    sb_px = sb_mm * ppm

    plate_margin = 50
    plate_w = canvas_w - 2 * plate_margin
    plate_h = canvas_h - 2 * plate_margin - 40
    plate_x = float(plate_margin)
    plate_y = 70.0

    # Place holes
    holes = []
    for i, d_mm in enumerate(diameters):
        d_px = d_mm * ppm
        cx, cy = plate_x + plate_w / 2, plate_y + plate_h / 2
        for _ in range(100):
            cx_lo = plate_x + d_px / 2 + 15
            cx_hi = plate_x + plate_w - d_px / 2 - 15
            cy_lo = plate_y + d_px / 2 + 15
            cy_hi = plate_y + plate_h - d_px / 2 - 15
            if cx_lo >= cx_hi:
                cx_hi = cx_lo + 1
            if cy_lo >= cy_hi:
                cy_hi = cy_lo + 1
            cx = float(rng.uniform(cx_lo, cx_hi))
            cy = float(rng.uniform(cy_lo, cy_hi))
            ok = all(
                np.sqrt((cx - h["cx"]) ** 2 + (cy - h["cy"]) ** 2) > (d_px + h["d_px"]) / 2 + 8
                for h in holes
            )
            if ok:
                break

        holes.append({
            "label": f"H{i + 1}", "d_mm": d_mm,
            "d_px": round(d_px, 1),
            "cx": round(cx, 1), "cy": round(cy, 1),
        })

    # Compute ground truth based on hop level
    spec = {}
    gt = 0.0

    if hops == 2:
        gt = max(diameters) - min(diameters)
        spec_text = ""

    elif hops >= 4:
        # Diameter spec
        diam_min = round(float(rng.uniform(8, 13)), 1)
        diam_max = round(diam_min + float(rng.uniform(4, 8)), 1)
        spec["diam_min"] = diam_min
        spec["diam_max"] = diam_max

        diam_violations = 0.0
        for d in diameters:
            if d < diam_min:
                diam_violations += diam_min - d
            elif d > diam_max:
                diam_violations += d - diam_max

        gt = diam_violations
        spec_text = f"All hole diameters must be between {diam_min} mm and {diam_max} mm."

    if hops >= 6:
        # Spacing spec
        min_spacing = round(float(rng.uniform(12, 25)), 1)
        spec["min_spacing"] = min_spacing

        spacing_violations = 0.0
        for i in range(len(holes)):
            for j in range(i + 1, len(holes)):
                dist_px = np.sqrt(
                    (holes[i]["cx"] - holes[j]["cx"]) ** 2 +
                    (holes[i]["cy"] - holes[j]["cy"]) ** 2
                )
                dist_mm = dist_px / ppm
                if dist_mm < min_spacing:
                    spacing_violations += min_spacing - dist_mm

        gt += spacing_violations
        spec_text += f" Minimum spacing between any two holes must be {min_spacing} mm."

    if hops >= 8:
        # Edge distance spec
        min_edge = round(float(rng.uniform(5, 12)), 1)
        spec["min_edge"] = min_edge

        edge_violations = 0.0
        for h in holes:
            # Distance to each edge in mm
            left = (h["cx"] - plate_x) / ppm
            right = (plate_x + plate_w - h["cx"]) / ppm
            top = (h["cy"] - plate_y) / ppm
            bottom = (plate_y + plate_h - h["cy"]) / ppm
            min_dist = min(left, right, top, bottom)
            if min_dist < min_edge:
                edge_violations += min_edge - min_dist

        gt += edge_violations
        spec_text += f" Minimum distance from any hole center to any edge must be {min_edge} mm."

    gt = round(gt, 2)

    return {
        "idx": idx,
        "hops": hops,
        "n_holes": N_HOLES,
        "holes": holes,
        "diameters_mm": diameters,
        "gt_answer": gt,
        "spec": spec,
        "spec_text": spec_text,
        "scale_bar_mm": sb_mm,
        "sb_px": round(sb_px, 1),
        "ppm": round(ppm, 4),
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "plate_x": plate_x,
        "plate_y": plate_y,
        "plate_w": plate_w,
        "plate_h": plate_h,
    }


def render_image(sample: dict[str, Any], path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpl_patches

    cw, ch = sample["canvas_w"], sample["canvas_h"]
    fig, ax = plt.subplots(1, 1, figsize=(cw / 150, ch / 150), dpi=150)
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.axis("off")

    rect = mpl_patches.Rectangle(
        (sample["plate_x"], sample["plate_y"]),
        sample["plate_w"], sample["plate_h"],
        linewidth=1.5, edgecolor="black", facecolor="#F0F0EA", zorder=1,
    )
    ax.add_patch(rect)

    for h in sample["holes"]:
        circle = mpl_patches.Circle(
            (h["cx"], h["cy"]), h["d_px"] / 2,
            linewidth=1.0, edgecolor="black", facecolor="#333333", zorder=2,
        )
        ax.add_patch(circle)
        ax.text(h["cx"], h["cy"] + h["d_px"] / 2 + 10, h["label"],
                ha="center", va="bottom", fontsize=6, fontweight="bold", zorder=3)

    sb_x = sample["plate_x"]
    sb_y = sample["plate_y"] - 35
    sb_len = sample["sb_px"]
    ax.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color="black", linewidth=2, zorder=5)
    ax.plot([sb_x, sb_x], [sb_y - 5, sb_y + 5], color="black", linewidth=1.5, zorder=5)
    ax.plot([sb_x + sb_len, sb_x + sb_len], [sb_y - 5, sb_y + 5], color="black", linewidth=1.5, zorder=5)
    ax.text(sb_x + sb_len / 2, sb_y - 10, f"{sample['scale_bar_mm']} mm",
            ha="center", va="top", fontsize=7, zorder=5)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def generate_dataset(hops: int, n_train: int = 800, n_test: int = 200, seed: int = 50) -> None:
    rng = np.random.default_rng(seed + hops)
    out = Path(dataset_dir(hops))

    for split in ["train", "test"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    for split, n in [("train", n_train), ("test", n_test)]:
        samples = [generate_sample(rng, i, hops) for i in range(n)]

        with open(out / split / "metadata.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        for i, s in enumerate(samples):
            render_image(s, str(out / split / f"image_{s['idx']:04d}.png"))
            if (i + 1) % 100 == 0:
                print(f"  {split}: {i + 1}/{n}")

        gts = [s["gt_answer"] for s in samples]
        n_zero = sum(1 for g in gts if g == 0)
        print(f"\n{hops}-hop {split}: N={n}, answer range={min(gts):.1f}-{max(gts):.1f}, "
              f"mean={np.mean(gts):.1f}, zeros={n_zero}")

    print(f"Done. Output in {out}/")


# ========================
# UTILS
# ========================

def parse_number(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    m = re.search(r'[\d]+\.?\d*', text)
    return float(m.group(0)) if m else None


def make_prompt(hops: int, spec_text: str = "") -> str:
    cfg = HOP_CONFIGS[hops]
    system = cfg["system"].format(spec=spec_text) if "{spec}" in cfg["system"] else cfg["system"]
    return system + "\n\n" + cfg["question"]


def load_samples(hops: int, split: str) -> list[dict[str, Any]]:
    with open(Path(dataset_dir(hops)) / split / "metadata.jsonl") as f:
        return [json.loads(l) for l in f]


# ========================
# BASELINE
# ========================

def run_baseline(hops: int) -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"\n=== {hops}-hop Baseline ===\n")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    samples = load_samples(hops, "test")
    maes: list[float] = []

    for i, s in enumerate(samples):
        path = str(Path(dataset_dir(hops)) / "test" / f"image_{s['idx']:04d}.png")
        image = Image.open(path).convert("RGB")
        prompt = make_prompt(hops, s.get("spec_text", ""))
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        raw = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = parse_number(raw)
        if pred is not None:
            maes.append(abs(pred - s["gt_answer"]))

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/200] MAE={np.mean(maes):.2f}")

    gts = [s["gt_answer"] for s in samples]
    mean_guess = np.mean([abs(g - np.mean(gts)) for g in gts])

    print(f"\n{hops}-hop baseline MAE: {np.mean(maes):.2f}")
    print(f"Mean guess MAE: {mean_guess:.2f}")

    os.makedirs(f"results_hop{hops}", exist_ok=True)
    with open(f"results_hop{hops}/baseline.json", "w") as f:
        json.dump({"hops": hops, "mae": round(np.mean(maes), 2),
                    "mean_guess": round(mean_guess, 2)}, f, indent=2)

    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()


# ========================
# SFT
# ========================

def run_sft(hops: int) -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output = ckpt_dir(hops)
    print(f"\n=== {hops}-hop SFT ===\n")

    samples = load_samples(hops, "train")
    total_steps = len(samples) * NUM_EPOCHS
    print(f"Train: {len(samples)}, steps: {total_steps}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )

    if os.path.exists(WARMSTART_CKPT):
        print("Merging warmstart checkpoint...")
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
            path = str(Path(dataset_dir(hops)) / "train" / f"image_{s['idx']:04d}.png")
            image = Image.open(path).convert("RGB")

            prompt = make_prompt(hops, s.get("spec_text", ""))
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": str(s["gt_answer"])},
            ]

            text = processor.apply_chat_template(messages, tokenize=False)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            model.train()
            outputs = model(**inputs, labels=inputs["input_ids"].clone())
            loss = outputs.loss

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
    print(f"\n✓ {hops}-hop SFT complete. Saved to {final}")


# ========================
# EVAL
# ========================

def run_eval(hops: int) -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt = os.path.join(ckpt_dir(hops), "final")
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}")
        return

    print(f"\n=== {hops}-hop Eval ===\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )

    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()

    model = PeftModel.from_pretrained(model, ckpt)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    samples = load_samples(hops, "test")
    maes: list[float] = []

    for i, s in enumerate(samples):
        path = str(Path(dataset_dir(hops)) / "test" / f"image_{s['idx']:04d}.png")
        image = Image.open(path).convert("RGB")
        prompt = make_prompt(hops, s.get("spec_text", ""))
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        raw = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = parse_number(raw)
        if pred is not None:
            maes.append(abs(pred - s["gt_answer"]))

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/200] MAE={np.mean(maes):.2f}")

    result = {
        "hops": hops,
        "mae": round(np.mean(maes), 2),
        "median_ae": round(np.median(maes), 2),
        "within_1mm": round(np.mean([e < 1 for e in maes]) * 100, 1),
        "within_2mm": round(np.mean([e < 2 for e in maes]) * 100, 1),
        "within_5mm": round(np.mean([e < 5 for e in maes]) * 100, 1),
        "parsed": len(maes),
    }

    print(f"\n{hops}-hop SFT MAE: {result['mae']}mm")
    for k, v in result.items():
        print(f"  {k}: {v}")

    os.makedirs(f"results_hop{hops}", exist_ok=True)
    with open(f"results_hop{hops}/sft.json", "w") as f:
        json.dump(result, f, indent=2)

    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()


# ========================
# SUMMARY
# ========================

def run_summary() -> None:
    print(f"\n{'=' * 70}")
    print(f"  Hop Scaling Experiment: SFT MAE vs Complexity")
    print(f"{'=' * 70}\n")

    print(f"{'Hops':<6} {'Baseline':>10} {'Mean Guess':>12} {'SFT MAE':>10} {'Within 2mm':>12} {'Within 5mm':>12}")
    print("-" * 65)

    for hops in [2, 4, 6, 8]:
        bl_path = f"results_hop{hops}/baseline.json"
        sft_path = f"results_hop{hops}/sft.json"

        bl_mae = "—"
        mean_guess = "—"
        sft_mae = "—"
        w2 = "—"
        w5 = "—"

        if os.path.exists(bl_path):
            with open(bl_path) as f:
                bl = json.load(f)
            bl_mae = f"{bl['mae']:.2f}"
            mean_guess = f"{bl['mean_guess']:.2f}"

        if os.path.exists(sft_path):
            with open(sft_path) as f:
                sft = json.load(f)
            sft_mae = f"{sft['mae']:.2f}"
            w2 = f"{sft['within_2mm']:.1f}%"
            w5 = f"{sft['within_5mm']:.1f}%"

        print(f"{hops:<6} {bl_mae:>10} {mean_guess:>12} {sft_mae:>10} {w2:>12} {w5:>12}")

    print("-" * 65)
    print("\nIf SFT MAE increases with hops: SFT breaks at higher complexity.")
    print("The crossover point is where RL becomes relevant.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--hops", type=int, choices=[2, 4, 6, 8], default=2)
    args = parser.parse_args()

    if args.generate:
        generate_dataset(args.hops)
    if args.baseline:
        run_baseline(args.hops)
    if args.sft:
        run_sft(args.hops)
    if args.eval:
        run_eval(args.hops)
    if args.summary:
        run_summary()

    if not any([args.generate, args.baseline, args.sft, args.eval, args.summary]):
        parser.print_help()


if __name__ == "__main__":
    main()
