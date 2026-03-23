"""Task 6: 2-hop spatial measurement.

Question: "What is the diameter difference between the largest and
smallest hole in mm?"

The model must: (1) measure all hole diameters, (2) find max and min,
(3) subtract. Two spatial measurements + comparison + arithmetic.

Three experiments:
  --generate          Generate dataset (~2 min)
  --sft               SFT from multi-task checkpoint (~30 min)
  --sft-scratch       SFT from base model (~30 min)
  --grpo              GRPO from multi-task checkpoint (~60 min)
  --grpo-scratch      GRPO from base model (~60 min)
  --eval TAG          Evaluate a trained model

Usage:
    python3 task6_2hop.py --generate
    python3 task6_2hop.py --sft
    python3 task6_2hop.py --grpo
    python3 task6_2hop.py --eval sft
    python3 task6_2hop.py --eval grpo
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
MULTITASK_CKPT: str = "checkpoints_multitask/final"
DATASET_DIR: str = "dataset_task6"
LORA_RANK: int = 64
LORA_ALPHA: int = 128
LEARNING_RATE: float = 5e-6
NUM_EPOCHS: int = 3
LOG_EVERY: int = 10
SAVE_EVERY: int = 99999
NUM_GENERATIONS: int = 4
MAX_NEW_TOKENS: int = 64
KL_BETA: float = 0.1
REWARD_CLIP_MIN: float = -5.0

SCALE_BAR_VALUES: list[int] = [5, 10, 15, 20, 25, 30, 40, 50]
DIAMETER_RANGE: tuple[float, float] = (3.0, 30.0)
N_HOLES_RANGE: tuple[int, int] = (3, 6)

QUESTION: str = "What is the diameter difference between the largest and smallest hole in mm?"
SYSTEM: str = (
    "You are measuring holes in a technical drawing. "
    "Use the scale bar to determine hole diameters. "
    "Respond with ONLY the difference in mm as a number, nothing else."
)


# ========================
# GENERATOR
# ========================

def generate_dataset(n_train: int = 800, n_test: int = 200, seed: int = 47) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpl_patches

    rng = np.random.default_rng(seed)

    def gen_sample(rng: np.random.Generator, idx: int) -> dict[str, Any]:
        n_holes = int(rng.integers(*N_HOLES_RANGE))
        diameters = [round(float(rng.uniform(*DIAMETER_RANGE)), 2) for _ in range(n_holes)]
        sb_mm = int(rng.choice(SCALE_BAR_VALUES))
        canvas_w = int(rng.integers(500, 1000))
        canvas_h = int(rng.integers(400, 800))

        max_diam = max(diameters)
        ppm_min = max(12 / min(diameters), 30 / sb_mm)
        ppm_max = min(0.35 * canvas_w / max_diam, 0.35 * canvas_w / sb_mm)

        if ppm_min >= ppm_max:
            canvas_w = int(canvas_w * 1.5)
            canvas_h = int(canvas_h * 1.5)
            ppm_max = min(0.35 * canvas_w / max_diam, 0.35 * canvas_w / sb_mm)

        ppm = float(rng.uniform(ppm_min, min(ppm_max, ppm_min * 3)))
        sb_px = sb_mm * ppm

        plate_margin = 50
        plate_w = canvas_w - 2 * plate_margin
        plate_h = canvas_h - 2 * plate_margin - 40
        plate_x = float(plate_margin)
        plate_y = 70.0

        holes = []
        for i, d_mm in enumerate(diameters):
            d_px = d_mm * ppm
            for _ in range(50):
                cx = plate_x + rng.uniform(d_px / 2 + 10, plate_w - d_px / 2 - 10)
                cy = plate_y + rng.uniform(d_px / 2 + 10, plate_h - d_px / 2 - 10)
                ok = True
                for h in holes:
                    dist = np.sqrt((cx - h["cx"]) ** 2 + (cy - h["cy"]) ** 2)
                    if dist < (d_px + h["d_px"]) / 2 + 8:
                        ok = False
                        break
                if ok:
                    break
            holes.append({
                "label": f"H{i + 1}",
                "d_mm": d_mm,
                "d_px": round(d_px, 1),
                "cx": round(float(cx), 1),
                "cy": round(float(cy), 1),
            })

        gt_diff = round(max(diameters) - min(diameters), 2)

        return {
            "idx": idx,
            "n_holes": n_holes,
            "holes": holes,
            "diameters_mm": diameters,
            "max_diam": round(max(diameters), 2),
            "min_diam": round(min(diameters), 2),
            "gt_diff_mm": gt_diff,
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

    def render(sample: dict[str, Any], path: str) -> None:
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
                    ha="center", va="bottom", fontsize=7, fontweight="bold", zorder=3)

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

    out = Path(DATASET_DIR)
    for split in ["train", "test"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    n = {"train": n_train, "test": n_test}
    for split in ["train", "test"]:
        samples = [gen_sample(rng, i) for i in range(n[split])]

        with open(out / split / "metadata.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        for i, s in enumerate(samples):
            render(s, str(out / split / f"image_{s['idx']:04d}.png"))
            if (i + 1) % 100 == 0:
                print(f"  {split}: {i + 1}/{len(samples)}")

        diffs = [s["gt_diff_mm"] for s in samples]
        print(f"\n{split}: N={len(samples)}, diff range={min(diffs):.1f}-{max(diffs):.1f}mm, "
              f"mean={np.mean(diffs):.1f}mm")

    print(f"\nDone. Output in {DATASET_DIR}/")


# ========================
# SHARED UTILS
# ========================

def parse_number(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    m = re.search(r'[\d]+\.?\d*', text)
    return float(m.group(0)) if m else None


def load_samples(split: str = "train") -> list[dict[str, Any]]:
    with open(Path(DATASET_DIR) / split / "metadata.jsonl") as f:
        return [json.loads(l) for l in f]


def make_prompt() -> str:
    return SYSTEM + "\n\n" + QUESTION


def infer(model: Any, processor: Any, image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": make_prompt()},
    ]}]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
    return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ========================
# SFT TRAINING
# ========================

def run_sft(from_multitask: bool = True) -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    tag = "sft" if from_multitask else "sft_scratch"
    output_dir = f"checkpoints_task6_{tag}"
    print(f"\n=== Task 6 SFT ({'from multitask' if from_multitask else 'from scratch'}) ===\n")

    samples = load_samples("train")
    total_steps = len(samples) * NUM_EPOCHS
    print(f"Train: {len(samples)}, epochs: {NUM_EPOCHS}, steps: {total_steps}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )

    if from_multitask and os.path.exists(MULTITASK_CKPT):
        print(f"Merging multitask checkpoint...")
        model = PeftModel.from_pretrained(model, MULTITASK_CKPT)
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

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.jsonl")
    running_loss: list[float] = []
    global_step = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        rng = np.random.default_rng(42 + epoch)
        for idx in rng.permutation(len(samples)):
            global_step += 1
            s = samples[idx]
            image_path = str(Path(DATASET_DIR) / "train" / f"image_{s['idx']:04d}.png")
            image = Image.open(image_path).convert("RGB")

            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": make_prompt()},
                ]},
                {"role": "assistant", "content": str(s["gt_diff_mm"])},
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
                spm = global_step / (elapsed / 60)
                eta = (total_steps - global_step) / spm if spm > 0 else 0

                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": global_step, "loss": round(avg, 4)}) + "\n")
                print(f"  Step {global_step}/{total_steps} | loss={avg:.4f} | ETA={eta:.0f}m")

    final = os.path.join(output_dir, "final")
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"\n✓ SFT {tag} complete. Saved to {final}")


# ========================
# GRPO TRAINING
# ========================

def run_grpo(from_multitask: bool = True) -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    tag = "grpo" if from_multitask else "grpo_scratch"
    output_dir = f"checkpoints_task6_{tag}"
    print(f"\n=== Task 6 GRPO ({'from multitask' if from_multitask else 'from scratch'}) ===\n")

    samples = load_samples("train")
    print(f"Train: {len(samples)}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )

    if from_multitask and os.path.exists(MULTITASK_CKPT):
        print(f"Merging multitask checkpoint...")
        model = PeftModel.from_pretrained(model, MULTITASK_CKPT)
        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Creating reference model...")
    ref_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if from_multitask and os.path.exists(MULTITASK_CKPT):
        ref_model = PeftModel.from_pretrained(ref_model, MULTITASK_CKPT)
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
    model.print_trainable_parameters()

    a = torch.cuda.memory_allocated() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {a:.1f}GB / {t:.0f}GB")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE,
    )

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "log.jsonl")

    running_reward: list[float] = []
    running_loss: list[float] = []
    global_step = 0
    start_time = time.time()

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(samples))

    for idx in indices:
        global_step += 1
        s = samples[idx]
        image_path = str(Path(DATASET_DIR) / "train" / f"image_{s['idx']:04d}.png")
        gt = s["gt_diff_mm"]

        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": make_prompt()},
        ]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        # Generate
        completions: list[str] = []
        gen_ids_list: list[torch.Tensor] = []
        model.eval()
        for _ in range(NUM_GENERATIONS):
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True, temperature=0.7, top_p=0.9,
                )
            gids = out[0, input_len:].clone()
            gen_ids_list.append(gids)
            completions.append(processor.decode(gids, skip_special_tokens=True).strip())
            del out
        torch.cuda.empty_cache()

        # Rewards — continuous based on error
        rewards: list[float] = []
        for c in completions:
            pred = parse_number(c)
            if pred is None:
                rewards.append(REWARD_CLIP_MIN)
            else:
                rewards.append(max(-abs(pred - gt) / max(gt, 1.0), REWARD_CLIP_MIN))
        running_reward.extend(rewards)

        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in rewards]

        # Per-completion backward with KL
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

            kl = (tlp - ref_tlp).sum()
            loss = (-adv * tlp.sum() + KL_BETA * kl) / NUM_GENERATIONS
            loss.backward()
            total_loss += loss.item()

            del out, ref_out, logits, lp, ref_lp, tlp, ref_tlp, loss, full_ids
            torch.cuda.empty_cache()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running_loss.append(total_loss)

        del inputs, gen_ids_list
        torch.cuda.empty_cache()
        gc.collect()

        if global_step % LOG_EVERY == 0:
            avg_r = np.mean(running_reward[-LOG_EVERY * NUM_GENERATIONS:])
            avg_l = np.mean(running_loss[-LOG_EVERY:])
            elapsed = time.time() - start_time
            spm = global_step / (elapsed / 60) if elapsed > 0 else 0
            eta = (len(samples) - global_step) / spm if spm > 0 else 0

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": global_step, "reward": round(avg_r, 4),
                    "loss": round(avg_l, 4), "sample": completions[0][:80],
                }) + "\n")

            print(f"  Step {global_step}/{len(samples)} | "
                  f"reward={avg_r:.3f} | loss={avg_l:.4f} | ETA={eta:.0f}m | "
                  f"gt={gt:.1f} | '{completions[0][:50]}'")

    final = os.path.join(output_dir, "final")
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"\n✓ GRPO {tag} complete. Saved to {final}")


# ========================
# EVALUATION
# ========================

def run_eval(tag: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt = f"checkpoints_task6_{tag}/final"
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}")
        return

    print(f"\n=== Task 6 Eval: {tag} ===\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )

    if "scratch" not in tag and os.path.exists(MULTITASK_CKPT):
        model = PeftModel.from_pretrained(model, MULTITASK_CKPT)
        model = model.merge_and_unload()

    model = PeftModel.from_pretrained(model, ckpt)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    samples = load_samples("test")
    maes: list[float] = []
    parsed = 0

    for i, s in enumerate(samples):
        image_path = str(Path(DATASET_DIR) / "test" / f"image_{s['idx']:04d}.png")
        raw = infer(model, processor, image_path)
        pred = parse_number(raw)
        if pred is not None:
            parsed += 1
            maes.append(abs(pred - s["gt_diff_mm"]))

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/200] MAE={np.mean(maes):.2f}mm (parsed {parsed}/{i + 1})")

    print(f"\n=== {tag} Results ===")
    print(f"MAE: {np.mean(maes):.2f}mm")
    print(f"Median AE: {np.median(maes):.2f}mm")
    print(f"Within 1mm: {np.mean([e < 1 for e in maes]) * 100:.1f}%")
    print(f"Within 2mm: {np.mean([e < 2 for e in maes]) * 100:.1f}%")
    print(f"Within 5mm: {np.mean([e < 5 for e in maes]) * 100:.1f}%")
    print(f"Parsed: {parsed}/{len(samples)}")


def run_baseline() -> None:
    """Evaluate base model without any training."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"\n=== Task 6 Baseline (no training) ===\n")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    samples = load_samples("test")
    maes: list[float] = []

    for i, s in enumerate(samples):
        image_path = str(Path(DATASET_DIR) / "test" / f"image_{s['idx']:04d}.png")
        raw = infer(model, processor, image_path)
        pred = parse_number(raw)
        if pred is not None:
            maes.append(abs(pred - s["gt_diff_mm"]))

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/200] MAE={np.mean(maes):.2f}mm")

    diffs = [s["gt_diff_mm"] for s in samples]
    mean_guess = np.mean([abs(d - np.mean(diffs)) for d in diffs])

    print(f"\nBaseline MAE: {np.mean(maes):.2f}mm")
    print(f"Mean guess MAE: {mean_guess:.2f}mm")

    del model
    torch.cuda.empty_cache()
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--sft-scratch", action="store_true")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--grpo-scratch", action="store_true")
    parser.add_argument("--eval", type=str)
    args = parser.parse_args()

    if args.generate:
        generate_dataset()
    if args.baseline:
        run_baseline()
    if args.sft:
        run_sft(from_multitask=True)
    if args.sft_scratch:
        run_sft(from_multitask=False)
    if args.grpo:
        run_grpo(from_multitask=True)
    if args.grpo_scratch:
        run_grpo(from_multitask=False)
    if args.eval:
        run_eval(args.eval)

    if not any([args.generate, args.baseline, args.sft, args.sft_scratch,
                args.grpo, args.grpo_scratch, args.eval]):
        parser.print_help()


if __name__ == "__main__":
    main()
