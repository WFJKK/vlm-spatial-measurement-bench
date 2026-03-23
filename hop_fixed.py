"""Fixed hop scaling: answer-only loss masking + SAPO-style GRPO.

Two fixes:
1. SFT: mask prompt tokens, only compute loss on answer tokens
2. GRPO: replace hard clipping with SAPO soft sigmoid gate

Reuses existing datasets from hop_scaling.py.

Usage:
    python3 hop_fixed.py --sft --hops 4
    python3 hop_fixed.py --sft --hops 6
    python3 hop_fixed.py --grpo --hops 4
    python3 hop_fixed.py --grpo --hops 6
    python3 hop_fixed.py --eval sft --hops 4
    python3 hop_fixed.py --eval grpo --hops 4
    python3 hop_fixed.py --summary
"""

from __future__ import annotations

import argparse
import gc
import json
import math
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

# GRPO config
NUM_GENERATIONS: int = 4
MAX_NEW_TOKENS: int = 64
KL_BETA: float = 0.2  # higher than before (was 0.1)
REWARD_CLIP_MIN: float = -3.0  # tighter clip (was -5.0)
# SAPO temperatures
TAU_POS: float = 1.0
TAU_NEG: float = 1.05  # asymmetric: more conservative on negative advantages

HOP_CONFIGS = {
    4: {
        "question": "What is the total amount by which hole diameters are out of spec in mm? If all comply, answer 0.",
        "system": "You are inspecting a technical drawing. {spec} Use the scale bar to measure all hole diameters and sum up any violations. Respond with ONLY the total out-of-spec amount in mm as a number.",
    },
    6: {
        "question": "What is the total non-compliance in mm (diameter violations + spacing violations)? If all comply, answer 0.",
        "system": "You are inspecting a technical drawing. {spec} Use the scale bar to measure all hole diameters and distances between holes. Sum up all violations. Respond with ONLY the total in mm as a number.",
    },
}


def dataset_dir(hops: int) -> str:
    return f"dataset_hop{hops}"


def ckpt_dir(hops: int, method: str) -> str:
    return f"checkpoints_hop{hops}_{method}_fixed"


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
# SFT WITH ANSWER-ONLY LOSS
# ========================

def run_sft(hops: int) -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output = ckpt_dir(hops, "sft")
    print(f"\n=== {hops}-hop SFT (answer-only loss) ===\n")

    samples = load_samples(hops, "train")
    total_steps = len(samples) * NUM_EPOCHS
    print(f"Train: {len(samples)}, steps: {total_steps}")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if os.path.exists(WARMSTART_CKPT):
        print("Merging warmstart...")
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
            answer = str(s["gt_answer"])

            # Tokenize prompt (user turn) separately to find where answer starts
            user_msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            prompt_text = processor.apply_chat_template(
                user_msgs, add_generation_prompt=True, tokenize=False,
            )
            prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # Tokenize full (prompt + answer)
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

            # Create labels: -100 for prompt tokens, real ids for answer tokens
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
    print(f"\n✓ {hops}-hop SFT (fixed) complete. Saved to {final}")


# ========================
# SAPO-STYLE GRPO
# ========================

def sapo_gate(ratio: torch.Tensor, advantage: float) -> torch.Tensor:
    """SAPO soft sigmoid gate instead of hard clipping."""
    tau = TAU_POS if advantage > 0 else TAU_NEG
    return torch.sigmoid(tau * (ratio - 1.0))


def run_grpo(hops: int) -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output = ckpt_dir(hops, "grpo")
    sft_fixed_path = ckpt_dir(hops, "sft") + "/final"
    sft_orig_path = f"checkpoints_hop{hops}_sft/final"

    # Prefer fixed SFT, fall back to original
    sft_path = sft_fixed_path if os.path.exists(sft_fixed_path) else sft_orig_path

    print(f"\n=== {hops}-hop GRPO-SAPO (from {sft_path}) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"KL_BETA={KL_BETA}, TAU_POS={TAU_POS}, TAU_NEG={TAU_NEG}")

    if not os.path.exists(sft_path):
        print(f"ERROR: No SFT checkpoint. Run --sft first.")
        return

    samples = load_samples(hops, "train")
    print(f"Train: {len(samples)}")

    # Build policy: base + warmstart + SFT merged
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, sft_path)
    model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Reference = same merged model, frozen
    print("Creating reference model...")
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

    # LoRA on policy
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

    os.makedirs(output, exist_ok=True)
    log_path = os.path.join(output, "log.jsonl")

    running_reward: list[float] = []
    running_loss: list[float] = []
    global_step = 0
    start_time = time.time()

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(samples))

    for idx in indices:
        global_step += 1
        s = samples[idx]
        gt = s["gt_answer"]
        path = str(Path(dataset_dir(hops)) / "train" / f"image_{s['idx']:04d}.png")

        image = Image.open(path).convert("RGB")
        prompt = make_prompt(hops, s.get("spec_text", ""))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
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

        # Continuous reward
        rewards: list[float] = []
        for c in completions:
            pred = parse_number(c)
            if pred is None:
                rewards.append(REWARD_CLIP_MIN)
            else:
                error = abs(pred - gt)
                rewards.append(max(-error / max(gt, 1.0), REWARD_CLIP_MIN))
        running_reward.extend(rewards)

        mean_r = np.mean(rewards)
        std_r = np.std(rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in rewards]

        # Skip if all advantages are zero (vanishing advantages)
        if all(abs(a) < 0.01 for a in advantages):
            del inputs, gen_ids_list
            torch.cuda.empty_cache()
            continue

        # SAPO backward
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
                ref_logits = ref_out.logits[0, input_len - 1:-1, :]
                ref_lp = torch.log_softmax(ref_logits, dim=-1)
                ref_tlp = ref_lp[range(len(gids)), gids]

            # Token-level importance ratios
            with torch.no_grad():
                log_ratio = tlp - ref_tlp
                ratio = torch.exp(log_ratio)

            # SAPO soft gate per token
            gate = sapo_gate(ratio.detach(), adv)

            # Gated policy loss
            policy_loss = -(adv * gate * tlp).sum() / NUM_GENERATIONS
            kl = log_ratio.sum()
            kl_loss = KL_BETA * kl / NUM_GENERATIONS
            loss = policy_loss + kl_loss
            loss.backward()
            total_loss += loss.item()

            del out, ref_out, logits, ref_logits, lp, ref_lp
            del tlp, ref_tlp, ratio, gate, loss, full_ids
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
            spm = global_step / (elapsed / 60) if elapsed > 0 else 1
            eta = (len(samples) - global_step) / spm if spm > 0 else 0

            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": global_step, "reward": round(avg_r, 4),
                    "loss": round(avg_l, 4), "sample": completions[0][:60],
                }) + "\n")

            print(f"  Step {global_step}/{len(samples)} | "
                  f"reward={avg_r:.3f} | ETA={eta:.0f}m | "
                  f"gt={gt:.1f} | '{completions[0][:40]}'")

    final = os.path.join(output, "final")
    model.save_pretrained(final)
    processor.save_pretrained(final)
    print(f"\n✓ {hops}-hop GRPO-SAPO complete. Saved to {final}")


# ========================
# EVAL
# ========================

def run_eval(hops: int, method: str) -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt = os.path.join(ckpt_dir(hops, method), "final")
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}")
        return

    print(f"\n=== {hops}-hop {method} (fixed) Eval ===\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()

    # For GRPO eval: also load the SFT checkpoint it was warmstarted from
    if method == "grpo":
        sft_fixed = ckpt_dir(hops, "sft") + "/final"
        sft_orig = f"checkpoints_hop{hops}_sft/final"
        sft_path = sft_fixed if os.path.exists(sft_fixed) else sft_orig
        if os.path.exists(sft_path):
            model = PeftModel.from_pretrained(model, sft_path)
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
        "hops": hops, "method": f"{method}_fixed",
        "mae": round(np.mean(maes), 2),
        "median_ae": round(np.median(maes), 2),
        "within_2mm": round(np.mean([e < 2 for e in maes]) * 100, 1),
        "within_5mm": round(np.mean([e < 5 for e in maes]) * 100, 1),
        "parsed": len(maes),
    }

    print(f"\n{hops}-hop {method} (fixed) MAE: {result['mae']}mm")

    os.makedirs(f"results_hop{hops}", exist_ok=True)
    with open(f"results_hop{hops}/{method}_fixed.json", "w") as f:
        json.dump(result, f, indent=2)

    del model
    torch.cuda.empty_cache()
    gc.collect()


# ========================
# SUMMARY
# ========================

def run_summary() -> None:
    print(f"\n{'=' * 80}")
    print(f"  Hop Scaling: Original SFT vs Fixed SFT vs GRPO-SAPO")
    print(f"{'=' * 80}\n")

    print(f"{'Hops':<6} {'Mean Guess':>12} {'SFT (orig)':>12} {'SFT (fixed)':>13} {'GRPO-SAPO':>12} {'Best':>8}")
    print("-" * 66)

    for hops in [4, 6]:
        mg = "—"
        sft_orig = "—"
        sft_fixed = "—"
        grpo_sapo = "—"
        best = "—"

        bl_path = f"results_hop{hops}/baseline.json"
        orig_path = f"results_hop{hops}/sft.json"
        fixed_path = f"results_hop{hops}/sft_fixed.json"
        grpo_path = f"results_hop{hops}/grpo_fixed.json"

        values = {}

        if os.path.exists(bl_path):
            with open(bl_path) as f:
                mg = f"{json.load(f)['mean_guess']:.2f}"

        if os.path.exists(orig_path):
            with open(orig_path) as f:
                v = json.load(f)["mae"]
            sft_orig = f"{v:.2f}"
            values["SFT orig"] = v

        if os.path.exists(fixed_path):
            with open(fixed_path) as f:
                v = json.load(f)["mae"]
            sft_fixed = f"{v:.2f}"
            values["SFT fixed"] = v

        if os.path.exists(grpo_path):
            with open(grpo_path) as f:
                v = json.load(f)["mae"]
            grpo_sapo = f"{v:.2f}"
            values["GRPO-SAPO"] = v

        if values:
            best = min(values, key=values.get)

        print(f"{hops:<6} {mg:>12} {sft_orig:>12} {sft_fixed:>13} {grpo_sapo:>12} {best:>8}")

    print("-" * 66)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", action="store_true")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--eval", type=str, choices=["sft", "grpo"])
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--hops", type=int, choices=[4, 6], default=4)
    args = parser.parse_args()

    if args.sft:
        run_sft(args.hops)
    if args.grpo:
        run_grpo(args.hops)
    if args.eval:
        run_eval(args.hops, args.eval)
    if args.summary:
        run_summary()

    if not any([args.sft, args.grpo, args.eval, args.summary]):
        parser.print_help()


if __name__ == "__main__":
    main()
