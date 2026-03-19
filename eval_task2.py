"""Task 2 evaluation: distance between two points.

Runs Qwen2.5-VL-7B baseline on Task 2 images (no training).
Tests whether the spatial decoding bottleneck generalizes from
diameter estimation (Task 1) to distance estimation (Task 2).

Task 1 bottleneck: probe 4.31mm, output 7.72mm, gap 3.41mm.
If Task 2 shows a similar gap: bottleneck is task-independent.
If Task 2 gap is larger: distance is harder to decode than size.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

DATASET_DIR: str = "dataset_task2"
RESULTS_DIR: str = "results/task2_baseline"
EMBEDDINGS_DIR: str = "embeddings"
MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"

SYSTEM_PROMPT: str = (
    "You are measuring a technical drawing. "
    "Use the scale bar to determine the distance between the two points. "
    "Respond with ONLY the distance in mm as a number, nothing else."
)
USER_PROMPT: str = "What is the distance between P1 and P2 in mm?"


def parse_number(text: str) -> float | None:
    """Extract a numeric value from model output."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = re.search(r'(\d+\.?\d*)', text)
    return float(match.group(1)) if match else None


def load_model() -> tuple[Any, Any]:
    """Load Qwen2.5-VL-7B for inference."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading {MODEL_ID}...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    return model, processor


def run_inference(model: Any, processor: Any, image_path: str) -> str:
    """Run single-image inference and return decoded text."""
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def smoke_test(model: Any, processor: Any) -> bool:
    """Verify inference works on one image before full evaluation."""
    meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
    with open(meta_path) as f:
        sample = json.loads(f.readline())

    image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")
    print(f"  Smoke test: image {sample['idx']}, gt={sample['distance_mm']:.1f}mm")

    try:
        text = run_inference(model, processor, image_path)
        pred = parse_number(text)
        print(f"  Output: '{text}' -> {pred}mm")
        print(f"  Smoke test PASSED")
        return True
    except Exception as e:
        print(f"  Smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_eval() -> dict[str, Any] | None:
    """Run baseline evaluation on test set + matched pairs."""
    model, processor = load_model()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n--- Smoke Test ---")
    if not smoke_test(model, processor):
        print("ABORTING: smoke test failed")
        return None

    meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
    with open(meta_path) as f:
        samples = [json.loads(l) for l in f]

    print(f"\n=== Task 2 Baseline: Qwen2.5-VL-7B ===")
    print(f"Test samples: {len(samples)}")

    results: list[dict[str, Any]] = []
    maes: list[float] = []
    for i, sample in enumerate(samples):
        image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")
        try:
            text = run_inference(model, processor, image_path)
        except Exception as e:
            print(f"    Image {sample['idx']}: ERROR {e}")
            text = ""

        pred = parse_number(text)
        gt = sample["distance_mm"]
        ae = abs(pred - gt) if pred else None

        results.append({
            "idx": sample["idx"],
            "gt_mm": gt,
            "predicted_mm": pred,
            "raw_output": text,
            "ae_mm": ae,
        })

        if ae is not None:
            maes.append(ae)

        if (i + 1) % 10 == 0:
            current_mae = np.mean(maes) if maes else float("nan")
            print(f"    [{i + 1}/{len(samples)}] MAE so far: {current_mae:.2f}mm "
                  f"(parsed: {len(maes)}/{i + 1})")

    with open(os.path.join(RESULTS_DIR, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    valid_aes = [r["ae_mm"] for r in results if r["ae_mm"] is not None]
    metrics: dict[str, Any] = {
        "task": "distance_between_points",
        "model": MODEL_ID,
        "n_total": len(results),
        "n_parsed": len(valid_aes),
        "parse_rate": len(valid_aes) / len(results),
        "mae_mm": round(np.mean(valid_aes), 3) if valid_aes else None,
        "median_ae_mm": round(np.median(valid_aes), 3) if valid_aes else None,
        "within_1mm": round(np.mean([a < 1 for a in valid_aes]), 3) if valid_aes else None,
        "within_2mm": round(np.mean([a < 2 for a in valid_aes]), 3) if valid_aes else None,
        "within_5mm": round(np.mean([a < 5 for a in valid_aes]), 3) if valid_aes else None,
    }

    print(f"\n=== Task 2 Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    matched_dir = Path(DATASET_DIR) / "test_matched"
    if matched_dir.exists():
        with open(matched_dir / "metadata.jsonl") as f:
            pairs = [json.loads(l) for l in f]

        print(f"\n=== Matched Pair Diagnostic ({len(pairs)} pairs) ===")

        pair_results: list[dict[str, Any]] = []
        for j, pm in enumerate(pairs):
            pid = pm["pair_id"]
            img_a = str(matched_dir / f"pair_{pid:03d}_a.png")
            img_b = str(matched_dir / f"pair_{pid:03d}_b.png")

            try:
                text_a = run_inference(model, processor, img_a)
                text_b = run_inference(model, processor, img_b)
            except Exception as e:
                print(f"    Pair {pid}: ERROR {e}")
                continue

            pred_a = parse_number(text_a)
            pred_b = parse_number(text_b)

            pair_results.append({
                "pair_id": pid,
                "gt_a": pm["dist_a"], "gt_b": pm["dist_b"],
                "pred_a": pred_a, "pred_b": pred_b,
                "same_answer": pred_a == pred_b if (pred_a and pred_b) else None,
            })

            if (j + 1) % 10 == 0:
                print(f"    [{j + 1}/{len(pairs)}] pairs done")

        valid = [p for p in pair_results if p["pred_a"] and p["pred_b"]]
        if valid:
            gt_d = [p["gt_a"] - p["gt_b"] for p in valid]
            pr_d = [p["pred_a"] - p["pred_b"] for p in valid]
            corr = float(np.corrcoef(gt_d, pr_d)[0, 1]) if len(valid) > 2 else 0.0
            same = sum(1 for p in valid if p["same_answer"]) / len(valid)

            if corr > 0.5:
                interp = "USES scale bar"
            elif corr > 0.2:
                interp = "PARTIALLY uses scale bar"
            else:
                interp = "IGNORES scale bar"

            metrics["matched_pair"] = {
                "n_valid": len(valid),
                "diff_correlation": round(corr, 4),
                "frac_same_answer": round(same, 3),
                "interpretation": interp,
            }
            print(f"  Correlation: {corr:.4f}, Same: {same:.1%}, {interp}")

        with open(os.path.join(RESULTS_DIR, "matched_pairs.json"), "w") as f:
            json.dump(pair_results, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Results saved to {RESULTS_DIR}/")

    del model
    torch.cuda.empty_cache()
    return metrics


def run_probe() -> None:
    """Extract vision encoder embeddings and train linear probe for Task 2."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    from transformers import AutoModelForImageTextToText, AutoProcessor

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    out_path = os.path.join(EMBEDDINGS_DIR, "task2_baseline.npy")

    if os.path.exists(out_path):
        print(f"Embeddings already exist at {out_path}, loading...")
        X = np.load(out_path)
    else:
        print(f"Loading {MODEL_ID} for probing...")
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        model.eval()

        merger_module = None
        for path in ["visual.merger", "model.visual.merger"]:
            mod = model
            try:
                for attr in path.split("."):
                    mod = getattr(mod, attr)
                merger_module = mod
                print(f"  Found merger at: {path}")
                break
            except AttributeError:
                continue

        if merger_module is None:
            print("  ERROR: Could not find vision merger. Aborting probe.")
            return

        meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
        with open(meta_path) as f:
            samples = [json.loads(l) for l in f]

        embeddings: list[np.ndarray] = []
        for i, sample in enumerate(samples):
            image_path = str(Path(DATASET_DIR) / "test" / f"image_{sample['idx']:04d}.png")
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"},
            ]}]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

            vision_output: dict[str, torch.Tensor] = {}

            def hook_fn(module: Any, input: Any, output: Any) -> None:
                if isinstance(output, tuple):
                    vision_output["features"] = output[0].detach().cpu().float()
                elif isinstance(output, torch.Tensor):
                    vision_output["features"] = output.detach().cpu().float()

            handle = merger_module.register_forward_hook(hook_fn)
            with torch.no_grad():
                try:
                    model.generate(**inputs, max_new_tokens=1)
                except Exception:
                    pass
            handle.remove()

            if "features" in vision_output:
                features = vision_output["features"]
                if features.dim() == 3:
                    pooled = features[0].mean(dim=0)
                elif features.dim() == 2:
                    pooled = features.mean(dim=0)
                else:
                    pooled = features.flatten()
                embeddings.append(pooled.numpy())
            else:
                print(f"    Image {sample['idx']}: no features captured")
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[-1]))

            if (i + 1) % 50 == 0:
                print(f"    {i + 1}/{len(samples)}")

        X = np.stack(embeddings)
        np.save(out_path, X)
        print(f"  Saved embeddings: {X.shape}")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    meta_path = Path(DATASET_DIR) / "test" / "metadata.jsonl"
    with open(meta_path) as f:
        samples = [json.loads(l) for l in f]
    gt = np.array([s["distance_mm"] for s in samples])

    probe = Ridge(alpha=1.0)
    scores = cross_val_score(probe, X, gt, cv=5, scoring="neg_mean_absolute_error")
    mae = -scores.mean()
    mae_std = scores.std()

    probe.fit(X, gt)
    r2 = probe.score(X, gt)

    print(f"\n{'=' * 70}")
    print(f"  Task 2 Vision Encoder Probe (Qwen2.5-VL-7B)")
    print(f"{'=' * 70}")
    print(f"  Embedding dim: {X.shape[1]}")
    print(f"  Probe MAE: {mae:.2f}mm (±{mae_std:.2f})")
    print(f"  Probe R²: {r2:.3f}")
    print()
    print(f"  Task 1 comparison (diameter):")
    print(f"    Probe MAE: 4.31mm  R²=0.942")
    print(f"    Output MAE: 7.72mm")
    print(f"    Gap: 3.41mm")
    print()

    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        output_mae = metrics.get("mae_mm")
        if output_mae:
            gap = output_mae - mae
            print(f"  Task 2 (distance):")
            print(f"    Probe MAE: {mae:.2f}mm")
            print(f"    Output MAE: {output_mae:.2f}mm")
            print(f"    Gap: {gap:.2f}mm")
            print()
            if abs(gap - 3.41) < 1.5:
                print(f"  → Similar gap to Task 1: bottleneck is TASK-INDEPENDENT")
            elif gap > 3.41 + 1.5:
                print(f"  → Larger gap than Task 1: distance is HARDER to decode")
            else:
                print(f"  → Smaller gap than Task 1: distance may be EASIER to decode")

    probe_results = {
        "task": "distance_between_points",
        "model": MODEL_ID,
        "embedding_dim": int(X.shape[1]),
        "probe_mae_mm": round(mae, 2),
        "probe_mae_std": round(mae_std, 2),
        "probe_r2": round(r2, 3),
    }
    with open(os.path.join(EMBEDDINGS_DIR, "task2_probe_results.json"), "w") as f:
        json.dump(probe_results, f, indent=2)

    print(f"\n✓ Probe results saved")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 evaluation: distance between points")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or args.eval:
        run_eval()
    if args.all or args.probe:
        run_probe()
    if not (args.all or args.eval or args.probe):
        parser.print_help()


if __name__ == "__main__":
    main()
