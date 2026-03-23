"""GRPO for hop scaling experiment.

Runs GRPO warmstarted from SFT checkpoints at 4-hop and 6-hop.
Uses existing datasets from hop_scaling.py.

Usage:
    python3 hop_grpo.py --train --hops 4
    python3 hop_grpo.py --train --hops 6
    python3 hop_grpo.py --eval --hops 4
    python3 hop_grpo.py --eval --hops 6
    python3 hop_grpo.py --summary
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
LORA_RANK: int = 64
LORA_ALPHA: int = 128
LEARNING_RATE: float = 5e-6
NUM_GENERATIONS: int = 4
MAX_NEW_TOKENS: int = 64
KL_BETA: float = 0.1
REWARD_CLIP_MIN: float = -5.0
LOG_EVERY: int = 10

# Import question configs from hop_scaling
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


def sft_ckpt(hops: int) -> str:
    return f"checkpoints_hop{hops}_sft/final"


def grpo_ckpt_dir(hops: int) -> str:
    return f"checkpoints_hop{hops}_grpo"


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


def run_grpo(hops: int) -> None:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output = grpo_ckpt_dir(hops)
    sft_path = sft_ckpt(hops)

    print(f"\n=== {hops}-hop GRPO (warmstart from SFT) ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(sft_path):
        print(f"ERROR: SFT checkpoint not found at {sft_path}")
        print("Run hop_scaling.py --sft first.")
        return

    samples = load_samples(hops, "train")
    print(f"Train: {len(samples)}")

    # Load base + warmstart + SFT (merge both into base)
    print(f"Loading model + merging warmstart + SFT...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )

    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()

    model = PeftModel.from_pretrained(model, sft_path)
    model = model.merge_and_unload()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Reference model = same as starting policy (SFT checkpoint)
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

    # Apply LoRA on top
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

        # Generate N completions
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

        # Backward
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
    print(f"\n✓ {hops}-hop GRPO complete. Saved to {final}")


def run_eval(hops: int) -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    ckpt = os.path.join(grpo_ckpt_dir(hops), "final")
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}")
        return

    print(f"\n=== {hops}-hop GRPO Eval ===\n")

    # Load base + warmstart + SFT + GRPO
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    if os.path.exists(WARMSTART_CKPT):
        model = PeftModel.from_pretrained(model, WARMSTART_CKPT)
        model = model.merge_and_unload()
    if os.path.exists(sft_ckpt(hops)):
        model = PeftModel.from_pretrained(model, sft_ckpt(hops))
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
        "method": "grpo",
        "mae": round(np.mean(maes), 2),
        "median_ae": round(np.median(maes), 2),
        "within_1mm": round(np.mean([e < 1 for e in maes]) * 100, 1),
        "within_2mm": round(np.mean([e < 2 for e in maes]) * 100, 1),
        "within_5mm": round(np.mean([e < 5 for e in maes]) * 100, 1),
        "parsed": len(maes),
    }

    print(f"\n{hops}-hop GRPO MAE: {result['mae']}mm")
    for k, v in result.items():
        print(f"  {k}: {v}")

    os.makedirs(f"results_hop{hops}", exist_ok=True)
    with open(f"results_hop{hops}/grpo.json", "w") as f:
        json.dump(result, f, indent=2)

    del model
    torch.cuda.empty_cache()
    gc.collect()


def run_summary() -> None:
    print(f"\n{'=' * 75}")
    print(f"  Hop Scaling: SFT vs GRPO")
    print(f"{'=' * 75}\n")

    print(f"{'Hops':<6} {'Baseline':>10} {'Mean Guess':>12} {'SFT MAE':>10} {'GRPO MAE':>10} {'Winner':>8}")
    print("-" * 60)

    for hops in [2, 4, 6, 8]:
        bl_mae = "—"
        mg = "—"
        sft_mae = "—"
        grpo_mae = "—"
        winner = "—"

        bl_path = f"results_hop{hops}/baseline.json"
        sft_path = f"results_hop{hops}/sft.json"
        grpo_path = f"results_hop{hops}/grpo.json"

        if os.path.exists(bl_path):
            with open(bl_path) as f:
                bl = json.load(f)
            bl_mae = f"{bl['mae']:.2f}"
            mg = f"{bl['mean_guess']:.2f}"

        sft_val = None
        if os.path.exists(sft_path):
            with open(sft_path) as f:
                sft = json.load(f)
            sft_mae = f"{sft['mae']:.2f}"
            sft_val = sft['mae']

        grpo_val = None
        if os.path.exists(grpo_path):
            with open(grpo_path) as f:
                grpo = json.load(f)
            grpo_mae = f"{grpo['mae']:.2f}"
            grpo_val = grpo['mae']

        if sft_val and grpo_val:
            winner = "GRPO" if grpo_val < sft_val else "SFT"
        elif sft_val:
            winner = "SFT"

        print(f"{hops:<6} {bl_mae:>10} {mg:>12} {sft_mae:>10} {grpo_mae:>10} {winner:>8}")

    print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--hops", type=int, choices=[4, 6], default=4)
    args = parser.parse_args()

    if args.train:
        run_grpo(args.hops)
    if args.eval:
        run_eval(args.hops)
    if args.summary:
        run_summary()

    if not any([args.train, args.eval, args.summary]):
        parser.print_help()


if __name__ == "__main__":
    main()
