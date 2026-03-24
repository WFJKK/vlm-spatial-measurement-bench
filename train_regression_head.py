"""Regression head experiment: bypass text token decoding.

Instead of decoding the VLM's internal representation into text tokens
("42.3"), attach a regression head to the LM's last hidden state and
predict the diameter as a continuous scalar.

The LM still does the heavy lifting: it processes the instruction,
attends across vision patches (comparing scale bar to hole), and builds
an internal representation. We just change the final output step.

Variants:
  --mode base_head_only    : Freeze entire base VLM, train head only
  --mode base_lora_head    : LoRA-tune base VLM + train head jointly
  --mode sft3_head_only    : Freeze SFT-3epoch VLM, train head only

If sft3_head_only beats 1.62mm (the SFT-3epoch text result), the model
"knows" the answer more precisely than it can express in tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_DIR: str = "dataset"
RESULTS_DIR: str = "results"

# Same prompts as the rest of the repo
SYSTEM_PROMPT: str = (
    "You are measuring a hole in a technical drawing. "
    "Use the scale bar to determine the diameter. "
    "Respond with ONLY the diameter in mm as a number, nothing else."
)
USER_PROMPT: str = "What is the diameter of hole H1 in mm?"

# Training config
LEARNING_RATE_HEAD: float = 1e-3
LEARNING_RATE_LORA: float = 5e-6
NUM_EPOCHS: int = 3
LOG_EVERY: int = 10
SAVE_EVERY: int = 100
LORA_RANK: int = 64
LORA_ALPHA: int = 128


class RegressionHead(nn.Module):
    """Small MLP that maps LM hidden state to a scalar measurement."""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_dataset(split: str = "train") -> list[dict[str, Any]]:
    meta_path = Path(DATASET_DIR) / split / "metadata.jsonl"
    with open(meta_path) as f:
        return [json.loads(l) for l in f]


def load_model(mode: str):
    """Load model based on mode, return (model, processor, requires_lora)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if mode == "sft3_head_only":
        sft_path = "checkpoints_sft_3epoch/final"
        if not os.path.exists(sft_path):
            raise FileNotFoundError(
                f"SFT-3epoch checkpoint not found at {sft_path}. "
                "Run train_sft_3epoch.py first."
            )
        from peft import PeftModel
        print(f"Merging SFT-3epoch adapter from {sft_path}...")
        model = PeftModel.from_pretrained(model, sft_path)
        model = model.merge_and_unload()

    if mode == "base_lora_head":
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor


def get_hidden_dim(model) -> int:
    """Get the LM hidden dimension from model config."""
    config = model.config
    # Qwen2.5-VL stores it in text_config or directly
    if hasattr(config, "text_config"):
        return config.text_config.hidden_size
    if hasattr(config, "hidden_size"):
        return config.hidden_size
    raise ValueError("Cannot determine hidden_size from model config")


def extract_last_hidden_state(
    model,
    processor,
    image: Image.Image,
) -> torch.Tensor:
    """Forward pass to get the LM's hidden state at the last input token.

    This is the representation the model would normally use to predict
    the first output token. It contains the result of all cross-patch
    attention computation.
    """
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
    ]}]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model(**inputs, output_hidden_states=True)

    # Last layer's hidden states, last token position
    # hidden_states is a tuple of (n_layers+1,) each [batch, seq_len, hidden]
    last_layer = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
    last_token = last_layer[0, -1, :]       # [hidden_dim]

    return last_token


def train(mode: str, resume: bool = False) -> None:
    print(f"=== Regression Head Training: {mode} ===\n")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    samples = load_dataset("train")
    print(f"Train samples: {len(samples)}")

    model, processor = load_model(mode)
    hidden_dim = get_hidden_dim(model)
    print(f"LM hidden dimension: {hidden_dim}")

    # Create regression head
    head = RegressionHead(hidden_dim).to(model.device).to(torch.bfloat16)
    print(f"Regression head parameters: {sum(p.numel() for p in head.parameters()):,}")

    # Freeze model for head-only modes
    if mode in ("base_head_only", "sft3_head_only"):
        for param in model.parameters():
            param.requires_grad = False
        print("Model frozen, training head only")
        optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE_HEAD)
    else:
        # Joint training: LoRA params + head
        lora_params = [p for p in model.parameters() if p.requires_grad]
        head_params = list(head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": LEARNING_RATE_LORA},
            {"params": head_params, "lr": LEARNING_RATE_HEAD},
        ])
        print(f"Joint training: {sum(p.numel() for p in lora_params):,} LoRA + "
              f"{sum(p.numel() for p in head_params):,} head params")

    output_dir = f"checkpoints_regression_{mode}"
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training_log.jsonl")

    start_step = 0
    if resume and os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        if lines:
            start_step = json.loads(lines[-1])["step"]
            print(f"Resuming from step {start_step}")

            # Try to load head checkpoint
            head_ckpt = os.path.join(output_dir, f"head_checkpoint-{start_step}.pt")
            if os.path.exists(head_ckpt):
                head.load_state_dict(torch.load(head_ckpt, weights_only=True))
                print(f"Loaded head weights from {head_ckpt}")

    loss_fn = nn.MSELoss()
    global_step = 0
    running_loss: list[float] = []

    for epoch in range(NUM_EPOCHS):
        rng = np.random.default_rng(42 + epoch)
        indices = rng.permutation(len(samples))

        for idx in indices:
            global_step += 1
            if global_step <= start_step:
                continue

            sample = samples[idx]
            image_path = str(
                Path(DATASET_DIR) / "train" / f"image_{sample['idx']:04d}.png"
            )
            gt_mm = sample["diameter_mm"]

            image = Image.open(image_path).convert("RGB")

            # Forward pass to get hidden state
            if mode in ("base_head_only", "sft3_head_only"):
                with torch.no_grad():
                    h = extract_last_hidden_state(model, processor, image)
                # Only head is differentiable
                h = h.detach().requires_grad_(True)
            else:
                # LoRA mode: gradients flow through model
                h = extract_last_hidden_state(model, processor, image)

            # Regression head
            head.train()
            pred = head(h)
            target = torch.tensor(gt_mm, dtype=torch.bfloat16, device=pred.device)
            loss = loss_fn(pred, target)

            optimizer.zero_grad()
            loss.backward()
            if mode == "base_lora_head":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss.append(loss.item())

            if global_step % LOG_EVERY == 0:
                avg_loss = np.mean(running_loss[-LOG_EVERY:])
                mae = np.sqrt(avg_loss)  # Approximate MAE from MSE
                log_entry = {
                    "step": global_step,
                    "epoch": epoch,
                    "avg_mse_loss": round(float(avg_loss), 4),
                    "approx_mae": round(float(mae), 2),
                    "gt_mm": gt_mm,
                    "pred_mm": round(float(pred.item()), 2),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                print(
                    f"  Step {global_step} | epoch {epoch} | "
                    f"MSE={avg_loss:.4f} | ~MAE={mae:.2f}mm | "
                    f"pred={pred.item():.1f} gt={gt_mm:.1f}"
                )

            if global_step % SAVE_EVERY == 0:
                head_path = os.path.join(
                    output_dir, f"head_checkpoint-{global_step}.pt"
                )
                torch.save(head.state_dict(), head_path)
                if mode == "base_lora_head":
                    lora_path = os.path.join(
                        output_dir, f"lora_checkpoint-{global_step}"
                    )
                    model.save_pretrained(lora_path)
                print(f"  Saved checkpoint at step {global_step}")

    # Save final
    final_head = os.path.join(output_dir, "head_final.pt")
    torch.save(head.state_dict(), final_head)
    if mode == "base_lora_head":
        model.save_pretrained(os.path.join(output_dir, "lora_final"))
    print(f"\nTraining complete. Head saved to {final_head}")


def evaluate(mode: str) -> None:
    """Evaluate regression head on test set + matched pairs."""
    print(f"\n=== Evaluating Regression Head: {mode} ===\n")

    output_dir = f"checkpoints_regression_{mode}"
    head_path = os.path.join(output_dir, "head_final.pt")
    if not os.path.exists(head_path):
        raise FileNotFoundError(f"No trained head at {head_path}")

    model, processor = load_model(mode)

    if mode == "base_lora_head":
        lora_path = os.path.join(output_dir, "lora_final")
        if os.path.exists(lora_path):
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()

    hidden_dim = get_hidden_dim(model)
    head = RegressionHead(hidden_dim).to(model.device).to(torch.bfloat16)
    head.load_state_dict(torch.load(head_path, weights_only=True))
    head.eval()

    for param in model.parameters():
        param.requires_grad = False

    # --- Test set ---
    test_samples = load_dataset("test")
    results = []
    errors = []

    print(f"Running test set ({len(test_samples)} samples)...")
    for i, sample in enumerate(test_samples):
        image_path = str(
            Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png"
        )
        gt = sample["diameter_mm"]
        image = Image.open(image_path).convert("RGB")

        with torch.no_grad():
            h = extract_last_hidden_state(model, processor, image)
            pred = head(h).item()

        error = abs(pred - gt)
        errors.append(error)
        results.append({
            "idx": sample["idx"],
            "ground_truth_mm": gt,
            "predicted_mm": round(pred, 3),
            "error_mm": round(error, 3),
        })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(test_samples)}] MAE so far: {np.mean(errors):.2f}mm")

    mae = np.mean(errors)
    median_ae = np.median(errors)
    within_1 = np.mean([e < 1.0 for e in errors])
    within_2 = np.mean([e < 2.0 for e in errors])

    print(f"\n  Test Results:")
    print(f"  MAE:       {mae:.3f} mm")
    print(f"  Median AE: {median_ae:.3f} mm")
    print(f"  Within 1mm: {within_1:.1%}")
    print(f"  Within 2mm: {within_2:.1%}")

    # Save results
    results_dir = os.path.join(RESULTS_DIR, f"regression_{mode}")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- Matched pairs ---
    matched_dir = Path(DATASET_DIR) / "test_matched"
    meta_path = matched_dir / "metadata.jsonl"
    if meta_path.exists():
        with open(meta_path) as f:
            pairs = [json.loads(l) for l in f]

        print(f"\nMatched pair diagnostic ({len(pairs)} pairs)...")
        pred_diffs = []
        gt_diffs = []

        for pair in pairs:
            pid = pair["pair_id"]
            img_a = str(matched_dir / f"pair_{pid:03d}_a.png")
            img_b = str(matched_dir / f"pair_{pid:03d}_b.png")

            image_a = Image.open(img_a).convert("RGB")
            image_b = Image.open(img_b).convert("RGB")

            with torch.no_grad():
                h_a = extract_last_hidden_state(model, processor, image_a)
                h_b = extract_last_hidden_state(model, processor, image_b)
                pred_a = head(h_a).item()
                pred_b = head(h_b).item()

            gt_a, gt_b = pair["diam_a"], pair["diam_b"]
            pred_diffs.append(pred_a - pred_b)
            gt_diffs.append(gt_a - gt_b)

        from scipy.stats import pearsonr
        r, p = pearsonr(gt_diffs, pred_diffs)
        print(f"  Matched pair correlation: r = {r:.3f} (p = {p:.4f})")

        pair_results = {
            "n_pairs": len(pairs),
            "correlation_r": round(r, 4),
            "correlation_p": round(p, 6),
        }
        with open(os.path.join(results_dir, "matched_pair_results.json"), "w") as f:
            json.dump(pair_results, f, indent=2)

    # --- Comparison table ---
    print("\n" + "=" * 60)
    print("  COMPARISON: Regression Head vs Text Decoding")
    print("=" * 60)
    print(f"  {'Method':<30} {'MAE (mm)':>10}")
    print(f"  {'-'*30} {'-'*10}")

    comparisons = [
        ("Vision probe (linear)", 4.31),
        ("Vision probe (MLP 512x256)", 4.91),
        ("Baseline text (no training)", 7.72),
        ("SFT 1-epoch text", 3.08),
        ("SFT 3-epoch text", 1.62),
        ("GRPO answer-only text", 3.53),
        ("SFT->RL text", 2.29),
        (f"Regression head ({mode})", round(mae, 2)),
    ]
    for name, val in comparisons:
        marker = " <-- this run" if "Regression" in name else ""
        print(f"  {name:<30} {val:>10.2f}{marker}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train/evaluate regression head on VLM hidden states"
    )
    parser.add_argument(
        "--mode",
        choices=["base_head_only", "base_lora_head", "sft3_head_only"],
        default="sft3_head_only",
        help="Which variant to run",
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, just evaluate")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    args = parser.parse_args()

    if not args.eval_only:
        train(args.mode, resume=args.resume)
    evaluate(args.mode)
