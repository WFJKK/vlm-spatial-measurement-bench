"""Task 5 training experiment: does perception pretraining help compliance?

Two conditions:
  A) Train Task 5 from base Qwen2.5-VL-7B (no pretraining)
  B) Train Task 5 from Task 1 SFT-3epoch checkpoint (perception pretrained)

If B > A: perception pretraining helps grounded reasoning.
If B no-image ≈ 50%: model measures from image, not text shortcuts.
If B no-image > 60%: model learned text shortcuts despite pretraining.

Usage:
    python3 train_task5.py --condition A    # from scratch (~30 min)
    python3 train_task5.py --condition B    # from T1 checkpoint (~30 min)
    python3 train_task5.py --eval A         # evaluate condition A
    python3 train_task5.py --eval B         # evaluate condition B
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
T1_CHECKPOINT: str = "checkpoints_task1_sft3/final"
DATASET_DIR: str = "dataset_task5"
LORA_RANK: int = 64
LORA_ALPHA: int = 128
LEARNING_RATE: float = 5e-6
NUM_EPOCHS: int = 3
SAVE_EVERY: int = 200
LOG_EVERY: int = 10


def load_dataset(split: str = "train") -> list[dict[str, Any]]:
    meta_path = Path(DATASET_DIR) / split / "metadata.jsonl"
    with open(meta_path) as f:
        return [json.loads(l) for l in f]


def make_prompt(spec_text: str) -> str:
    return (
        "You are inspecting a technical drawing for compliance. "
        + spec_text
        + " Use the scale bar to measure the hole diameter and determine if it complies. "
        "Respond with ONLY PASS or FAIL, nothing else."
    )


def parse_label(text: str) -> str | None:
    text = text.strip().upper()
    if "PASS" in text:
        return "PASS"
    if "FAIL" in text:
        return "FAIL"
    return None


def train(condition: str, resume: bool = False) -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output_dir = f"checkpoints_task5_{condition}"
    print(f"=== Task 5 Training: Condition {condition} ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    samples = load_dataset("train")
    total_steps = len(samples) * NUM_EPOCHS
    print(f"Train samples: {len(samples)}, epochs: {NUM_EPOCHS}, total steps: {total_steps}")

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )

    if condition == "B":
        if not os.path.exists(T1_CHECKPOINT):
            print(f"ERROR: {T1_CHECKPOINT} not found. Download first.")
            return
        print(f"Merging Task 1 checkpoint from {T1_CHECKPOINT}...")
        model = PeftModel.from_pretrained(model, T1_CHECKPOINT)
        model = model.merge_and_unload()
        print("  Merged. Applying fresh LoRA on top.")

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

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.jsonl")

    start_step = 0
    if resume and os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        if lines:
            start_step = json.loads(lines[-1])["step"]
            print(f"Resuming from step {start_step}")

    global_step = 0
    running_loss: list[float] = []
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        rng = np.random.default_rng(42 + epoch)
        indices = rng.permutation(len(samples))

        for idx in indices:
            global_step += 1
            if global_step <= start_step:
                continue

            sample = samples[idx]
            image_path = str(Path(DATASET_DIR) / "train" / f"image_{sample['idx']:04d}.png")
            image = Image.open(image_path).convert("RGB")

            prompt = make_prompt(sample["spec_text"])
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]},
                {"role": "assistant", "content": sample["label"]},
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

            running_loss.append(loss.item())

            if global_step % LOG_EVERY == 0:
                avg_loss = np.mean(running_loss[-LOG_EVERY:])
                elapsed = time.time() - start_time
                actual_steps = global_step - start_step
                spm = actual_steps / (elapsed / 60) if elapsed > 0 else 0
                eta = (total_steps - global_step) / spm if spm > 0 else 0

                log_entry = {
                    "step": global_step, "epoch": epoch,
                    "avg_loss": round(avg_loss, 4),
                    "label": sample["label"],
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                print(f"  Step {global_step}/{total_steps} (epoch {epoch + 1}) | "
                      f"loss={avg_loss:.4f} | ETA={eta:.0f}m | {sample['label']}")

            if global_step % SAVE_EVERY == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_path)
                processor.save_pretrained(ckpt_path)

    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"\n✓ Condition {condition} complete. Saved to {final_path}")


def evaluate(condition: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt = f"checkpoints_task5_{condition}/final"
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}")
        return

    print(f"\n=== Task 5 Evaluation: Condition {condition} ===\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )

    if condition == "B":
        if os.path.exists(T1_CHECKPOINT):
            model = PeftModel.from_pretrained(model, T1_CHECKPOINT)
            model = model.merge_and_unload()

    model = PeftModel.from_pretrained(model, ckpt)
    model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    def infer(image_path: str, spec_text: str) -> str:
        image = Image.open(image_path).convert("RGB")
        prompt = make_prompt(spec_text)
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Normal test
    samples = load_dataset("test")
    correct = 0
    total = 0
    for i, s in enumerate(samples):
        image_path = str(Path(DATASET_DIR) / "test" / f"image_{s['idx']:04d}.png")
        resp = infer(image_path, s["spec_text"])
        pred = parse_label(resp)
        if pred is not None:
            total += 1
            if pred == s["label"]:
                correct += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(samples)}] acc={correct / max(total, 1) * 100:.1f}%")

    acc = correct / max(total, 1) * 100
    print(f"\nCondition {condition} accuracy: {acc:.1f}% ({correct}/{total})")

    # No-image ablation
    correct_ni = 0
    total_ni = 0
    for s in samples:
        image_path = str(Path(DATASET_DIR) / "test_no_image" / f"image_{s['idx']:04d}.png")
        resp = infer(image_path, s["spec_text"])
        pred = parse_label(resp)
        if pred is not None:
            total_ni += 1
            if pred == s["label"]:
                correct_ni += 1

    ni_acc = correct_ni / max(total_ni, 1) * 100
    print(f"No-image accuracy: {ni_acc:.1f}% (should be ~50%)")

    # Reasoning pairs
    with open(Path(DATASET_DIR) / "test_reasoning_pairs" / "metadata.jsonl") as f:
        pairs = [json.loads(l) for l in f]

    both_correct = 0
    spec_ignored = 0
    for pm in pairs:
        pid = pm["pair_id"]
        ps = pm["pass_sample"]
        fs = pm["fail_sample"]
        rp = parse_label(infer(
            str(Path(DATASET_DIR) / "test_reasoning_pairs" / f"pair_{pid:03d}_pass.png"),
            ps["spec_text"],
        ))
        rf = parse_label(infer(
            str(Path(DATASET_DIR) / "test_reasoning_pairs" / f"pair_{pid:03d}_fail.png"),
            fs["spec_text"],
        ))
        if rp == "PASS" and rf == "FAIL":
            both_correct += 1
        if rp == rf:
            spec_ignored += 1

    print(f"Reasoning pairs: {both_correct}/{len(pairs)} both correct")
    print(f"Spec ignored: {spec_ignored}/{len(pairs)}")

    print(f"\n{'=' * 50}")
    print(f"  Condition {condition}: acc={acc:.1f}% | no-img={ni_acc:.1f}% | pairs={both_correct}/{len(pairs)}")
    print(f"{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, choices=["A", "B"])
    parser.add_argument("--eval", type=str, choices=["A", "B"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.condition:
        train(args.condition, resume=args.resume)
    if args.eval:
        evaluate(args.eval)
    if not args.condition and not args.eval:
        parser.print_help()


if __name__ == "__main__":
    main()
