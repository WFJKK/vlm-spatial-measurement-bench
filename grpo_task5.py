"""Task 5 GRPO: does RL discover the two-stage output format?

Binary reward: +1 correct PASS/FAIL, -1 wrong. No format constraint.
The model can output anything — just "PASS", or "13.1mm PASS", or
a paragraph of reasoning. We only judge the final verdict.

Two conditions:
  A) GRPO from base Qwen2.5-VL-7B
  B) GRPO from Task 1 SFT-3epoch checkpoint (perception pretrained)

Key question: does the model start outputting measurements before
verdicts WITHOUT being told to? If yes, that's emergent grounded
reasoning — analogous to DeepSeek R1 discovering chain-of-thought.

Usage:
    python3 grpo_task5.py --condition A
    python3 grpo_task5.py --condition B
    python3 grpo_task5.py --eval A
    python3 grpo_task5.py --eval B
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
T1_CHECKPOINT: str = "checkpoints_task1_sft3/final"
DATASET_DIR: str = "dataset_task5"
LORA_RANK: int = 64
LORA_ALPHA: int = 128
LEARNING_RATE: float = 5e-6
NUM_GENERATIONS: int = 4
MAX_NEW_TOKENS: int = 128
NUM_EPOCHS: int = 1
LOG_EVERY: int = 5
SAVE_EVERY: int = 99999
KL_BETA: float = 0.1


def load_dataset(split: str = "train") -> list[dict[str, Any]]:
    meta_path = Path(DATASET_DIR) / split / "metadata.jsonl"
    with open(meta_path) as f:
        return [json.loads(l) for l in f]


def make_prompt(spec_text: str) -> str:
    return (
        "You are inspecting a technical drawing for compliance. "
        + spec_text
        + " Use the scale bar to measure the hole diameter and determine if it complies. "
        "Respond with your assessment."
    )


def parse_verdict(text: str) -> str | None:
    t = text.strip().upper()
    if "PASS" in t:
        return "PASS"
    if "FAIL" in t:
        return "FAIL"
    return None


def extract_measurement(text: str) -> float | None:
    """Try to find a measurement in mm from the output."""
    m = re.search(r'(\d+\.?\d*)\s*mm', text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def compute_reward(text: str, gt_label: str) -> float:
    verdict = parse_verdict(text)
    if verdict is None:
        return -1.0
    return 1.0 if verdict == gt_label else -1.0


def classify_output(text: str) -> str:
    """Classify what strategy the model uses."""
    has_number = bool(re.search(r'\d+\.?\d*\s*mm', text, re.IGNORECASE))
    has_verdict = parse_verdict(text) is not None
    has_reasoning = len(text) > 30

    if has_number and has_verdict and has_reasoning:
        return "measure+reason+verdict"
    elif has_number and has_verdict:
        return "measure+verdict"
    elif has_verdict and has_reasoning:
        return "reason+verdict"
    elif has_verdict:
        return "verdict_only"
    else:
        return "unparseable"


def grpo_step(
    model: Any,
    ref_model: Any,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    image_path: str,
    spec_text: str,
    gt_label: str,
) -> tuple[float, list[float], list[str]]:
    """One GRPO step with KL penalty."""
    image = Image.open(image_path).convert("RGB")
    prompt = make_prompt(spec_text)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    # Generate N completions
    completions: list[str] = []
    gen_ids_list: list[torch.Tensor] = []
    model.eval()
    for _ in range(NUM_GENERATIONS):
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True, temperature=0.7, top_p=0.9,
            )
        gen_ids = outputs[0, input_len:].clone()
        gen_ids_list.append(gen_ids)
        completions.append(processor.decode(gen_ids, skip_special_tokens=True).strip())
        del outputs
    torch.cuda.empty_cache()

    # Rewards and advantages
    rewards = [compute_reward(c, gt_label) for c in completions]
    mean_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]

    # Per-completion backward with KL penalty
    model.train()
    optimizer.zero_grad()
    total_loss_val = 0.0

    for gen_ids, advantage in zip(gen_ids_list, advantages):
        if len(gen_ids) > MAX_NEW_TOKENS:
            gen_ids = gen_ids[:MAX_NEW_TOKENS]

        full_ids = torch.cat([inputs["input_ids"][0], gen_ids]).unsqueeze(0)

        # Policy log probs
        out = model(
            input_ids=full_ids,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        logits = out.logits[0, input_len - 1:-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[range(len(gen_ids)), gen_ids]

        # Reference log probs for KL
        with torch.no_grad():
            ref_out = ref_model(
                input_ids=full_ids,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
            ref_logits = ref_out.logits[0, input_len - 1:-1, :]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            ref_token_log_probs = ref_log_probs[range(len(gen_ids)), gen_ids]

        kl = (token_log_probs - ref_token_log_probs).sum()
        policy_loss = -advantage * token_log_probs.sum() / NUM_GENERATIONS
        kl_loss = KL_BETA * kl / NUM_GENERATIONS
        loss = policy_loss + kl_loss
        loss.backward()
        total_loss_val += loss.item()

        del out, ref_out, logits, ref_logits, log_probs, ref_log_probs
        del token_log_probs, ref_token_log_probs, loss, full_ids
        torch.cuda.empty_cache()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    del inputs, gen_ids_list
    torch.cuda.empty_cache()
    gc.collect()

    return total_loss_val, rewards, completions


def train(condition: str) -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output_dir = f"checkpoints_grpo5_{condition}"
    print(f"=== Task 5 GRPO: Condition {condition} ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    samples = load_dataset("train")
    print(f"Train samples: {len(samples)}")

    # Load model
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )

    if condition == "B":
        if not os.path.exists(T1_CHECKPOINT):
            print(f"ERROR: {T1_CHECKPOINT} not found")
            return
        print(f"Merging Task 1 checkpoint...")
        model = PeftModel.from_pretrained(model, T1_CHECKPOINT)
        model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Create reference model (frozen copy before LoRA)
    print("Creating reference model...")
    ref_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if condition == "B" and os.path.exists(T1_CHECKPOINT):
        ref_model = PeftModel.from_pretrained(ref_model, T1_CHECKPOINT)
        ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # Apply LoRA to policy model
    model.gradient_checkpointing_enable()
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

    # Memory check
    a = torch.cuda.memory_allocated() / 1e9
    t = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {a:.1f}GB / {t:.0f}GB")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.jsonl")
    strategy_log_path = os.path.join(output_dir, "strategy_evolution.jsonl")

    global_step = 0
    running_reward: list[float] = []
    running_loss: list[float] = []
    strategy_counts: dict[str, int] = {}
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        rng = np.random.default_rng(42 + epoch)
        indices = rng.permutation(len(samples))

        for idx in indices:
            global_step += 1
            sample = samples[idx]
            image_path = str(Path(DATASET_DIR) / "train" / f"image_{sample['idx']:04d}.png")

            try:
                loss_val, rewards, completions = grpo_step(
                    model, ref_model, processor, optimizer,
                    image_path, sample["spec_text"], sample["label"],
                )
                running_reward.extend(rewards)
                running_loss.append(loss_val)

                # Track strategy evolution
                for c in completions:
                    strat = classify_output(c)
                    strategy_counts[strat] = strategy_counts.get(strat, 0) + 1

            except torch.cuda.OutOfMemoryError:
                print(f"  Step {global_step}: OOM, skipping")
                torch.cuda.empty_cache()
                gc.collect()
                optimizer.zero_grad()
                continue
            except Exception as e:
                print(f"  Step {global_step}: ERROR {e}")
                torch.cuda.empty_cache()
                continue

            if global_step % LOG_EVERY == 0:
                avg_r = np.mean(running_reward[-LOG_EVERY * NUM_GENERATIONS:])
                avg_l = np.mean(running_loss[-LOG_EVERY:])
                elapsed = time.time() - start_time
                spm = global_step / (elapsed / 60) if elapsed > 0 else 0
                eta = (len(samples) - global_step) / spm if spm > 0 else 0

                # Strategy distribution
                total_strat = sum(strategy_counts.values())
                strat_pcts = {k: round(v / total_strat * 100, 1)
                              for k, v in sorted(strategy_counts.items())}

                log_entry = {
                    "step": global_step,
                    "avg_reward": round(avg_r, 4),
                    "avg_loss": round(avg_l, 4),
                    "gt": sample["label"],
                    "strategies": strat_pcts,
                    "sample_output": completions[0][:120],
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

                # Strategy evolution snapshot
                with open(strategy_log_path, "a") as f:
                    f.write(json.dumps({"step": global_step, **strat_pcts}) + "\n")

                strat_str = " | ".join(f"{k}:{v}%" for k, v in strat_pcts.items())
                preview = completions[0][:80].replace('\n', ' ')
                print(f"  Step {global_step}/{len(samples)} | "
                      f"reward={avg_r:.3f} | ETA={eta:.0f}m | "
                      f"{strat_str}")
                print(f"    '{preview}'")

    final_path = os.path.join(output_dir, "final")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)

    print(f"\n✓ GRPO Condition {condition} complete. Saved to {final_path}")
    print(f"\nFinal strategy distribution:")
    total_strat = sum(strategy_counts.values())
    for k, v in sorted(strategy_counts.items()):
        print(f"  {k}: {v}/{total_strat} ({v/total_strat*100:.1f}%)")


def evaluate(condition: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt = f"checkpoints_grpo5_{condition}/final"
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}")
        return

    print(f"\n=== Task 5 GRPO Evaluation: Condition {condition} ===\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )

    if condition == "B" and os.path.exists(T1_CHECKPOINT):
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
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Normal test
    samples = load_dataset("test")
    correct = 0
    total = 0
    strategies: dict[str, int] = {}
    measurements_found = 0
    meas_errors: list[float] = []

    for i, s in enumerate(samples):
        image_path = str(Path(DATASET_DIR) / "test" / f"image_{s['idx']:04d}.png")
        raw = infer(image_path, s["spec_text"])
        pred = parse_verdict(raw)
        strat = classify_output(raw)
        strategies[strat] = strategies.get(strat, 0) + 1

        meas = extract_measurement(raw)
        if meas is not None:
            measurements_found += 1
            meas_errors.append(abs(meas - s["diameter_mm"]))

        if pred is not None:
            total += 1
            if pred == s["label"]:
                correct += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(samples)}] acc={correct / max(total, 1) * 100:.1f}%")

    acc = correct / max(total, 1) * 100

    print(f"\nCondition {condition} accuracy: {acc:.1f}% ({correct}/{total})")
    print(f"\nOutput strategy distribution:")
    for k, v in sorted(strategies.items()):
        print(f"  {k}: {v}/{len(samples)} ({v / len(samples) * 100:.1f}%)")

    print(f"\nMeasurements found in output: {measurements_found}/{len(samples)}")
    if meas_errors:
        print(f"Measurement MAE: {np.mean(meas_errors):.2f}mm")

    # No-image ablation
    correct_ni = 0
    total_ni = 0
    for s in samples:
        image_path = str(Path(DATASET_DIR) / "test_no_image" / f"image_{s['idx']:04d}.png")
        raw = infer(image_path, s["spec_text"])
        pred = parse_verdict(raw)
        if pred is not None:
            total_ni += 1
            if pred == s["label"]:
                correct_ni += 1
    ni_acc = correct_ni / max(total_ni, 1) * 100
    print(f"No-image accuracy: {ni_acc:.1f}%")

    # Reasoning pairs
    with open(Path(DATASET_DIR) / "test_reasoning_pairs" / "metadata.jsonl") as f:
        pairs = [json.loads(l) for l in f]

    both_correct = 0
    spec_ignored = 0
    for pm in pairs:
        pid = pm["pair_id"]
        ps = pm["pass_sample"]
        fs = pm["fail_sample"]
        rp = parse_verdict(infer(
            str(Path(DATASET_DIR) / "test_reasoning_pairs" / f"pair_{pid:03d}_pass.png"),
            ps["spec_text"],
        ))
        rf = parse_verdict(infer(
            str(Path(DATASET_DIR) / "test_reasoning_pairs" / f"pair_{pid:03d}_fail.png"),
            fs["spec_text"],
        ))
        if rp == "PASS" and rf == "FAIL":
            both_correct += 1
        if rp == rf:
            spec_ignored += 1

    print(f"Reasoning pairs: {both_correct}/{len(pairs)} both correct")
    print(f"Spec ignored: {spec_ignored}/{len(pairs)}")

    # Sample outputs
    print(f"\nSample outputs:")
    for s in samples[:10]:
        image_path = str(Path(DATASET_DIR) / "test" / f"image_{s['idx']:04d}.png")
        raw = infer(image_path, s["spec_text"])
        preview = raw[:100].replace('\n', ' ')
        print(f"  gt={s['label']} | '{preview}'")

    print(f"\n{'=' * 60}")
    print(f"  GRPO {condition}: acc={acc:.1f}% | no-img={ni_acc:.1f}% | "
          f"pairs={both_correct}/{len(pairs)} | meas_found={measurements_found}/200")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, choices=["A", "B"])
    parser.add_argument("--eval", type=str, choices=["A", "B"])
    args = parser.parse_args()

    if args.condition:
        train(args.condition)
    if args.eval:
        evaluate(args.eval)
    if not args.condition and not args.eval:
        parser.print_help()


if __name__ == "__main__":
    main()
