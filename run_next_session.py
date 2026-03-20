"""Task 4 probe + multi-task SFT training (Tasks 1+2+4).

Run in order:
    python3 run_next_session.py --probe       # 10 min, answers: can vision encoder see offsets?
    python3 run_next_session.py --train       # ~90 min, multi-task SFT 3 epochs
    python3 run_next_session.py --eval        # 30 min, evaluate on all 3 test sets
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
LORA_RANK: int = 64
LORA_ALPHA: int = 128
LEARNING_RATE: float = 5e-6
NUM_EPOCHS: int = 3
SAVE_EVERY: int = 300
LOG_EVERY: int = 10
OUTPUT_DIR: str = "checkpoints_multitask"

TASK_CONFIGS = {
    "task1": {
        "dataset_dir": "dataset_task1",
        "gt_key": "diameter_mm",
        "system": "You are measuring a hole in a technical drawing. Use the scale bar to determine the diameter. Respond with ONLY the diameter in mm as a number, nothing else.",
        "question": "What is the diameter of hole H1 in mm?",
    },
    "task2": {
        "dataset_dir": "dataset_task2",
        "gt_key": "distance_mm",
        "system": "You are measuring a technical drawing. Use the scale bar to determine the distance between the two points. Respond with ONLY the distance in mm as a number, nothing else.",
        "question": "What is the distance between P1 and P2 in mm?",
    },
    "task4": {
        "dataset_dir": "dataset_task4",
        "gt_key": "offset_mm",
        "system": "You are inspecting a technical drawing with two holes. Use the scale bar to determine the vertical offset between the centers of H1 and H2. Respond with ONLY the offset in mm as a number, nothing else.",
        "question": "What is the vertical offset between H1 and H2 in mm?",
    },
}


def parse_number(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    m = re.search(r'[\-]?\d+\.?\d*', text)
    return float(m.group(0)) if m else None


# ========================
# PROBE
# ========================

def run_probe() -> None:
    """Extract vision encoder embeddings and train linear probe for Task 4."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("=== Task 4 Vision Encoder Probe ===\n")
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    merger = model.model.visual.merger
    print("  Found merger module")

    cfg = TASK_CONFIGS["task4"]
    prompt = cfg["system"] + "\n\n" + cfg["question"]

    with open(Path(cfg["dataset_dir"]) / "test" / "metadata.jsonl") as f:
        samples = [json.loads(l) for l in f]

    embeddings: list[np.ndarray] = []
    for i, s in enumerate(samples):
        image_path = str(Path(cfg["dataset_dir"]) / "test" / f"image_{s['idx']:04d}.png")
        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        vision_out: dict[str, torch.Tensor] = {}

        def hook(module: Any, inp: Any, out: Any) -> None:
            if isinstance(out, tuple):
                vision_out["f"] = out[0].detach().cpu().float()
            else:
                vision_out["f"] = out.detach().cpu().float()

        h = merger.register_forward_hook(hook)
        with torch.no_grad():
            try:
                model.generate(**inputs, max_new_tokens=1)
            except Exception:
                pass
        h.remove()

        if "f" not in vision_out:
            print(f"  WARNING: hook did not fire for image {s['idx']}, using zeros")
            if embeddings:
                embeddings.append(np.zeros_like(embeddings[-1]))
            continue

        feat = vision_out["f"]
        if feat.dim() == 3:
            pooled = feat[0].mean(dim=0)
        elif feat.dim() == 2:
            pooled = feat.mean(dim=0)
        else:
            pooled = feat.flatten()
        embeddings.append(pooled.numpy())

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(samples)}")

    X = np.stack(embeddings)
    gt = np.array([s[cfg["gt_key"]] for s in samples])

    probe = Ridge(alpha=1.0)
    scores = cross_val_score(probe, X, gt, cv=5, scoring="neg_mean_absolute_error")
    mae = -scores.mean()
    mae_std = scores.std()

    probe.fit(X, gt)
    r2 = probe.score(X, gt)

    print(f"\n{'=' * 60}")
    print(f"  Task 4 Probe: MAE = {mae:.2f}mm (±{mae_std:.2f}), R² = {r2:.3f}")
    print(f"  Output MAE (baseline): 7.22mm")
    print(f"  Mean guess MAE: 1.85mm")
    print(f"{'=' * 60}")

    if mae > 1.7:
        print(f"\n  Probe MAE ({mae:.2f}) ≈ mean guess (1.85)")
        print(f"  → Vision encoder CANNOT see offsets. Resolution limit.")
        print(f"  → Training will NOT help. Skip multi-task training for Task 4.")
    else:
        print(f"\n  Probe MAE ({mae:.2f}) < mean guess (1.85)")
        print(f"  → Vision encoder CAN see offsets. Decoding bottleneck.")
        print(f"  → Training should help.")

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ========================
# GENERATE DATASETS
# ========================

def generate_datasets() -> None:
    """Generate all three task datasets if not already present."""
    for task, cfg in TASK_CONFIGS.items():
        if os.path.exists(cfg["dataset_dir"]):
            print(f"  {task}: dataset exists, skipping")
            continue
        if task == "task1":
            os.system("python3 generate_task1.py --n-train 1000 --n-test 200 --output-dir dataset_task1")
        elif task == "task2":
            os.system("python3 generate_task2.py --n-train 1000 --n-test 200 --output-dir dataset_task2")
        elif task == "task4":
            os.system("python3 generate_task4.py --n-train 1000 --n-test 200 --output-dir dataset_task4")


# ========================
# MULTI-TASK TRAINING
# ========================

def load_multitask_samples() -> list[dict[str, Any]]:
    """Load and mix training samples from all tasks."""
    all_samples: list[dict[str, Any]] = []

    for task_name, cfg in TASK_CONFIGS.items():
        meta_path = Path(cfg["dataset_dir"]) / "train" / "metadata.jsonl"
        with open(meta_path) as f:
            samples = [json.loads(l) for l in f]

        for s in samples:
            all_samples.append({
                "task": task_name,
                "image_path": str(Path(cfg["dataset_dir"]) / "train" / f"image_{s['idx']:04d}.png"),
                "gt_mm": s[cfg["gt_key"]],
                "system": cfg["system"],
                "question": cfg["question"],
            })

    print(f"  Total multi-task samples: {len(all_samples)}")
    for task_name in TASK_CONFIGS:
        count = sum(1 for s in all_samples if s["task"] == task_name)
        print(f"    {task_name}: {count}")

    return all_samples


def run_train(resume: bool = False) -> None:
    """Run multi-task SFT training."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("=== Multi-Task SFT Training (Tasks 1+2+4) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    generate_datasets()
    samples = load_multitask_samples()
    total_steps = len(samples) * NUM_EPOCHS
    print(f"Epochs: {NUM_EPOCHS}, total steps: {total_steps}")

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules="all-linear", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_DIR, "training_log.jsonl")

    start_step = 0
    if resume and os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        if lines:
            start_step = json.loads(lines[-1])["step"]
            print(f"Resuming from step {start_step}")

    global_step = 0
    running_loss: list[float] = []
    task_losses: dict[str, list[float]] = {t: [] for t in TASK_CONFIGS}
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        rng = np.random.default_rng(42 + epoch)
        indices = rng.permutation(len(samples))

        for idx in indices:
            global_step += 1
            if global_step <= start_step:
                continue

            sample = samples[idx]
            image = Image.open(sample["image_path"]).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": sample["system"] + "\n\n" + sample["question"]},
                ]},
                {"role": "assistant", "content": f"{sample['gt_mm']}"},
            ]

            text = processor.apply_chat_template(messages, tokenize=False)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            model.train()
            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            running_loss.append(loss_val)
            task_losses[sample["task"]].append(loss_val)

            if global_step % LOG_EVERY == 0:
                avg_loss = np.mean(running_loss[-LOG_EVERY:])
                elapsed = time.time() - start_time
                actual_steps = global_step - start_step
                spm = actual_steps / (elapsed / 60) if elapsed > 0 else 0
                eta = (total_steps - global_step) / spm if spm > 0 else 0

                log_entry = {
                    "step": global_step, "epoch": epoch,
                    "avg_loss": round(avg_loss, 4),
                    "task": sample["task"],
                    "gt_mm": sample["gt_mm"],
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                print(f"  Step {global_step}/{total_steps} (epoch {epoch + 1}) | "
                      f"loss={avg_loss:.4f} | ETA={eta:.0f}m | "
                      f"{sample['task']} gt={sample['gt_mm']:.1f}")

            if global_step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                processor.save_pretrained(ckpt_path)
                print(f"  Saved: {ckpt_path}")

    final_path = os.path.join(OUTPUT_DIR, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ Multi-task training complete. Saved to {final_path}")


# ========================
# EVALUATION
# ========================

def run_eval() -> None:
    """Evaluate multi-task model on all three test sets + matched pairs."""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt = os.path.join(OUTPUT_DIR, "final")
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}. Run --train first.")
        return

    print("=== Multi-Task Evaluation ===\n")
    print(f"Loading model + adapter from {ckpt}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, ckpt)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    def infer(image_path: str, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    print(f"\n{'Task':<10} {'MAE':>8} {'Baseline':>10} {'Single-task':>12} {'Matched r':>10}")
    print("-" * 55)

    baselines = {"task1": 7.72, "task2": 10.92, "task4": 7.22}
    single_task = {"task1": 1.62, "task2": 1.71, "task4": "n/a"}

    for task_name, cfg in TASK_CONFIGS.items():
        prompt = cfg["system"] + "\n\n" + cfg["question"]

        with open(Path(cfg["dataset_dir"]) / "test" / "metadata.jsonl") as f:
            samples = [json.loads(l) for l in f]

        maes: list[float] = []
        for s in samples:
            image_path = str(Path(cfg["dataset_dir"]) / "test" / f"image_{s['idx']:04d}.png")
            resp = infer(image_path, prompt)
            pred = parse_number(resp)
            if pred is not None:
                maes.append(abs(pred - s[cfg["gt_key"]]))

        mae = np.mean(maes) if maes else float("nan")

        # Matched pairs
        matched_dir = Path(cfg["dataset_dir"]) / "test_matched"
        corr = float("nan")
        if matched_dir.exists():
            with open(matched_dir / "metadata.jsonl") as f:
                pairs = [json.loads(l) for l in f]

            for pm in pairs:
                pid = pm["pair_id"]
                pm["pred_a"] = parse_number(infer(
                    str(matched_dir / f"pair_{pid:03d}_a.png"), prompt
                ))
                pm["pred_b"] = parse_number(infer(
                    str(matched_dir / f"pair_{pid:03d}_b.png"), prompt
                ))

            # Find the right gt keys for this task's matched pairs
            gt_key_a, gt_key_b = None, None
            for k in ["offset_a", "dist_a", "diam_a"]:
                if k in pairs[0]:
                    gt_key_a = k
                    gt_key_b = k.replace("_a", "_b")
                    break

            if gt_key_a is None:
                print(f"  {task_name}: could not find matched pair gt keys, skipping")
                continue

            valid = [p for p in pairs if p.get("pred_a") is not None and p.get("pred_b") is not None]
            if len(valid) > 2:
                gt_d = [p[gt_key_a] - p[gt_key_b] for p in valid]
                pr_d = [p["pred_a"] - p["pred_b"] for p in valid]
                corr = float(np.corrcoef(gt_d, pr_d)[0, 1])

        bl = baselines.get(task_name, "?")
        st = single_task.get(task_name, "?")
        print(f"  {task_name:<8} {mae:>8.2f} {bl:>10} {st:>12} {corr:>10.4f}")

    print("-" * 55)
    print(f"\nIf multi-task MAE ≈ single-task: calibration lock-in solved.")
    print(f"If Task 4 improves most: multi-task training covers the output range.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="Run Task 4 probe (10 min)")
    parser.add_argument("--train", action="store_true", help="Multi-task SFT training (~90 min)")
    parser.add_argument("--eval", action="store_true", help="Evaluate on all tasks (30 min)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all:
        run_probe()
        run_train(resume=args.resume)
        run_eval()
    else:
        if args.probe:
            run_probe()
        if args.train:
            run_train(resume=args.resume)
        if args.eval:
            run_eval()
        if not (args.probe or args.train or args.eval):
            parser.print_help()


if __name__ == "__main__":
    main()
