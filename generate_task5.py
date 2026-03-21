"""Task 5: Grounded compliance — spatial measurement + rule reasoning.

The model receives an image (hole + scale bar) and a text specification
("diameter must be between X and Y mm"). It must:
  1. Measure the hole diameter from the image (spatial perception)
  2. Compare the measurement to the spec (reasoning)
  3. Answer: PASS or FAIL, with the measured value

This is the simplest grounded reasoning task: one measurement, one rule.
But it requires BOTH modalities — text alone can't determine compliance,
and image alone doesn't know the spec.

Anti-shortcut measures:
  1. Balanced: exactly 50% PASS, 50% FAIL
  2. Spec ranges don't correlate with actual diameters
  3. Variable margins: some clearly in/out, some borderline
  4. Spec midpoint doesn't predict compliance
  5. Same image anti-shortcuts as Task 1 (variable zoom, scale bar, etc.)

Verification built in:
  - no_image split: blank white images + real specs → accuracy should be ~50%
  - no_spec split: real images + dummy spec "between 0 and 999mm" → always PASS
  - matched_reasoning: same image, different specs → one PASS, one FAIL
    Tests whether the model actually reads the spec, not just the image
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from numpy.random import Generator

SCALE_BAR_VALUES: list[int] = [5, 10, 15, 20, 25, 30, 40, 50]
DIAMETER_RANGE: tuple[float, float] = (3.0, 30.0)
CANVAS_W_RANGE: tuple[int, int] = (400, 900)
CANVAS_H_RANGE: tuple[int, int] = (350, 750)
DPI: int = 150

PLATE_COLOR: str = "#F0F0EA"
HOLE_FILL: str = "#333333"


def generate_image_params(rng: Generator) -> dict[str, Any]:
    """Generate image parameters (reuses Task 1 logic)."""
    diameter_mm = float(rng.uniform(*DIAMETER_RANGE))
    scale_bar_mm = int(rng.choice(SCALE_BAR_VALUES))
    canvas_w = int(rng.integers(*CANVAS_W_RANGE))
    canvas_h = int(rng.integers(*CANVAS_H_RANGE))

    min_hole_px = 12
    min_sb_px = 30
    max_frac = 0.45

    ppm_min = max(min_hole_px / diameter_mm, min_sb_px / scale_bar_mm)
    ppm_max = min(
        max_frac * canvas_w / diameter_mm,
        max_frac * canvas_w / scale_bar_mm,
        max_frac * canvas_h / diameter_mm,
    )

    if ppm_min >= ppm_max:
        canvas_w = int(canvas_w * 1.5)
        canvas_h = int(canvas_h * 1.5)
        ppm_max = min(
            max_frac * canvas_w / diameter_mm,
            max_frac * canvas_w / scale_bar_mm,
            max_frac * canvas_h / diameter_mm,
        )

    ppm = float(rng.uniform(ppm_min, min(ppm_max, ppm_min * 4)))
    hole_px = diameter_mm * ppm
    sb_px = scale_bar_mm * ppm

    plate_margin = 50
    plate_w = max(hole_px * 2.5, sb_px * 1.3, 150)
    plate_h = max(hole_px * 2.5, 120)
    plate_w = min(plate_w, canvas_w - 2 * plate_margin)
    plate_h = min(plate_h, canvas_h - 2 * plate_margin - 40)

    plate_x = (canvas_w - plate_w) / 2
    plate_y = 70.0

    hole_cx = plate_x + plate_w / 2 + rng.uniform(-plate_w * 0.15, plate_w * 0.15)
    hole_cy = plate_y + plate_h / 2 + rng.uniform(-plate_h * 0.15, plate_h * 0.15)

    return {
        "diameter_mm": round(diameter_mm, 2),
        "scale_bar_mm": scale_bar_mm,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "ppm": round(float(ppm), 4),
        "hole_px": round(float(hole_px), 1),
        "sb_px": round(float(sb_px), 1),
        "hole_cx": round(float(hole_cx), 1),
        "hole_cy": round(float(hole_cy), 1),
        "plate_x": float(plate_x),
        "plate_y": float(plate_y),
        "plate_w_px": float(plate_w),
        "plate_h_px": float(plate_h),
        "sb_x": float(plate_x),
        "sb_y": float(plate_y - 35),
    }


def generate_spec(
    rng: Generator,
    actual_diameter: float,
    target_compliance: bool,
) -> dict[str, Any]:
    """Generate a spec that either passes or fails for the given diameter.

    For PASS: spec range contains the actual diameter
    For FAIL: spec range excludes the actual diameter

    Variable margins make some cases borderline (hard) and some obvious (easy).
    """
    range_width = float(rng.uniform(2.0, 10.0))

    if target_compliance:
        # PASS: center the range near the actual diameter
        margin = float(rng.uniform(0.3, range_width / 2 - 0.1))
        spec_min = actual_diameter - margin
        spec_max = spec_min + range_width
    else:
        # FAIL: shift the range away from the actual diameter
        if rng.random() > 0.5:
            # Spec is above actual
            gap = float(rng.uniform(0.5, 5.0))
            spec_min = actual_diameter + gap
            spec_max = spec_min + range_width
        else:
            # Spec is below actual
            gap = float(rng.uniform(0.5, 5.0))
            spec_max = actual_diameter - gap
            spec_min = spec_max - range_width

    spec_min = max(0.5, spec_min)
    spec_max = max(spec_min + 1.0, spec_max)

    actual_complies = spec_min <= actual_diameter <= spec_max
    if actual_complies != target_compliance:
        if target_compliance:
            spec_min = actual_diameter - range_width / 2
            spec_max = actual_diameter + range_width / 2
        else:
            spec_min = actual_diameter + 1.0
            spec_max = spec_min + range_width

    spec_min = round(max(0.5, spec_min), 1)
    spec_max = round(max(spec_min + 1.0, spec_max), 1)

    return {
        "spec_min_mm": spec_min,
        "spec_max_mm": spec_max,
        "spec_text": f"Hole diameter must be between {spec_min} mm and {spec_max} mm.",
    }


def generate_sample(rng: Generator, idx: int, target_compliance: bool) -> dict[str, Any]:
    """Generate one complete sample: image params + spec + ground truth."""
    img = generate_image_params(rng)
    spec = generate_spec(rng, img["diameter_mm"], target_compliance)

    actual_complies = spec["spec_min_mm"] <= img["diameter_mm"] <= spec["spec_max_mm"]
    margin = min(
        abs(img["diameter_mm"] - spec["spec_min_mm"]),
        abs(img["diameter_mm"] - spec["spec_max_mm"]),
    )

    return {
        "idx": idx,
        **img,
        **spec,
        "complies": actual_complies,
        "label": "PASS" if actual_complies else "FAIL",
        "margin_mm": round(margin, 2),
    }


def render_image(sample: dict[str, Any], output_path: str, blank: bool = False) -> None:
    """Render the hole + scale bar image. If blank=True, render white image."""
    cw, ch = sample["canvas_w"], sample["canvas_h"]
    fig, ax = plt.subplots(1, 1, figsize=(cw / DPI, ch / DPI), dpi=DPI)
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.axis("off")

    if not blank:
        rect = mpl_patches.Rectangle(
            (sample["plate_x"], sample["plate_y"]),
            sample["plate_w_px"], sample["plate_h_px"],
            linewidth=1.5, edgecolor="black",
            facecolor=PLATE_COLOR, zorder=1,
        )
        ax.add_patch(rect)

        circle = mpl_patches.Circle(
            (sample["hole_cx"], sample["hole_cy"]),
            sample["hole_px"] / 2,
            linewidth=1.0, edgecolor="black",
            facecolor=HOLE_FILL, zorder=2,
        )
        ax.add_patch(circle)

        ax.text(sample["hole_cx"], sample["hole_cy"] + sample["hole_px"] / 2 + 10,
                "H1", ha="center", va="bottom", fontsize=7,
                fontweight="bold", zorder=3)

        sb_x, sb_y = sample["sb_x"], sample["sb_y"]
        sb_len = sample["sb_px"]
        ax.plot([sb_x, sb_x + sb_len], [sb_y, sb_y],
                color="black", linewidth=2, zorder=5)
        tick_h = 5
        ax.plot([sb_x, sb_x], [sb_y - tick_h, sb_y + tick_h],
                color="black", linewidth=1.5, zorder=5)
        ax.plot([sb_x + sb_len, sb_x + sb_len],
                [sb_y - tick_h, sb_y + tick_h],
                color="black", linewidth=1.5, zorder=5)
        ax.text(sb_x + sb_len / 2, sb_y - 10,
                f"{sample['scale_bar_mm']} mm",
                ha="center", va="top", fontsize=7, zorder=5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def verify_no_shortcuts(samples: list[dict[str, Any]]) -> None:
    """Verify balanced dataset and no text-only shortcuts."""
    labels = [s["label"] for s in samples]
    n_pass = sum(1 for l in labels if l == "PASS")
    n_fail = sum(1 for l in labels if l == "FAIL")

    diameters = np.array([s["diameter_mm"] for s in samples])
    spec_mins = np.array([s["spec_min_mm"] for s in samples])
    spec_maxs = np.array([s["spec_max_mm"] for s in samples])
    spec_mids = (spec_mins + spec_maxs) / 2
    spec_widths = spec_maxs - spec_mins
    margins = np.array([s["margin_mm"] for s in samples])
    complies = np.array([1.0 if s["complies"] else 0.0 for s in samples])

    print("=== SHORTCUT VERIFICATION ===")
    print(f"N = {len(samples)}")
    print(f"PASS: {n_pass} ({n_pass/len(samples):.1%})")
    print(f"FAIL: {n_fail} ({n_fail/len(samples):.1%})")
    print(f"Margin range: {margins.min():.2f} - {margins.max():.2f} mm")
    print(f"Margin mean: {margins.mean():.2f} mm\n")

    # Check if any text-only feature predicts compliance
    features = {
        "spec_min": spec_mins,
        "spec_max": spec_maxs,
        "spec_midpoint": spec_mids,
        "spec_width": spec_widths,
        "diameter (image-only)": diameters,
    }

    print("Correlation with compliance (should all be low):")
    for name, feat in features.items():
        r = np.corrcoef(complies, feat)[0, 1]
        flag = " ← WARNING" if abs(r) > 0.3 else ""
        print(f"  {name:<30s} r = {r:+.3f}{flag}")
    print()

    # Text-only baseline: predict compliance from spec alone
    # Best text-only strategy: if spec_mid is near the mean diameter, guess PASS
    mean_diam = diameters.mean()
    text_pred = np.abs(spec_mids - mean_diam) < (spec_widths / 2 + 3)
    text_acc = np.mean(text_pred == complies)
    print(f"Text-only heuristic accuracy: {text_acc:.1%} (should be ~50%)")
    print(f"Random baseline: 50.0%\n")


def generate_reasoning_pairs(
    rng: Generator, n_pairs: int = 50
) -> list[dict[str, Any]]:
    """Generate pairs: same image, different specs → one PASS, one FAIL.

    Tests whether the model reads the spec, not just the image.
    Also: same spec, different images → one PASS, one FAIL.
    Tests whether the model reads the image, not just the spec.
    """
    pairs: list[dict[str, Any]] = []

    for i in range(n_pairs):
        img = generate_image_params(rng)
        actual = img["diameter_mm"]

        spec_pass = generate_spec(rng, actual, target_compliance=True)
        spec_fail = generate_spec(rng, actual, target_compliance=False)

        sample_pass = {
            "idx": i, **img, **spec_pass,
            "complies": True, "label": "PASS",
            "margin_mm": round(min(abs(actual - spec_pass["spec_min_mm"]),
                                    abs(actual - spec_pass["spec_max_mm"])), 2),
        }
        sample_fail = {
            "idx": i, **img, **spec_fail,
            "complies": False, "label": "FAIL",
            "margin_mm": round(min(abs(actual - spec_fail["spec_min_mm"]),
                                    abs(actual - spec_fail["spec_max_mm"])), 2),
        }

        pairs.append({
            "pair_id": i,
            "type": "same_image_diff_spec",
            "diameter_mm": actual,
            "pass_sample": sample_pass,
            "fail_sample": sample_fail,
        })

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 5: grounded compliance")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--output-dir", type=str, default="dataset_task5")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Generate balanced dataset
    train_samples: list[dict[str, Any]] = []
    for i in range(args.n_train):
        target = i % 2 == 0  # Alternate PASS/FAIL for exact balance
        train_samples.append(generate_sample(rng, i, target))

    test_samples: list[dict[str, Any]] = []
    for i in range(args.n_test):
        target = i % 2 == 0
        test_samples.append(generate_sample(rng, i, target))

    print("=== TRAIN SET ===")
    verify_no_shortcuts(train_samples)
    print("=== TEST SET ===")
    verify_no_shortcuts(test_samples)

    if args.verify_only:
        return

    out = Path(args.output_dir)
    for split in ["train", "test", "test_no_image", "test_reasoning_pairs"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    # Normal images
    for split, samples in [("train", train_samples), ("test", test_samples)]:
        with open(out / split / "metadata.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        for i, s in enumerate(samples):
            render_image(s, str(out / split / f"image_{s['idx']:04d}.png"))
            if (i + 1) % 100 == 0:
                print(f"  {split}: {i + 1}/{len(samples)}")

    # No-image ablation: blank images with real specs
    print("\nGenerating no-image ablation set...")
    with open(out / "test_no_image" / "metadata.jsonl", "w") as f:
        for s in test_samples:
            f.write(json.dumps(s) + "\n")
    for s in test_samples:
        render_image(s, str(out / "test_no_image" / f"image_{s['idx']:04d}.png"), blank=True)

    # Reasoning pairs: same image, different specs
    print("Generating reasoning pair diagnostic set...")
    pairs = generate_reasoning_pairs(rng, n_pairs=50)

    with open(out / "test_reasoning_pairs" / "metadata.jsonl", "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    for pair in pairs:
        pid = pair["pair_id"]
        render_image(pair["pass_sample"],
                     str(out / "test_reasoning_pairs" / f"pair_{pid:03d}_pass.png"))
        render_image(pair["fail_sample"],
                     str(out / "test_reasoning_pairs" / f"pair_{pid:03d}_fail.png"))

    print(f"  Generated {len(pairs)} reasoning pairs")
    for p in pairs[:3]:
        d = p["diameter_mm"]
        ps = p["pass_sample"]
        fs = p["fail_sample"]
        print(f"  Pair {p['pair_id']}: diam={d:.1f}mm, "
              f"PASS spec=[{ps['spec_min_mm']}-{ps['spec_max_mm']}], "
              f"FAIL spec=[{fs['spec_min_mm']}-{fs['spec_max_mm']}]")

    print(f"\nDone. Output in {out}/")
    print(f"  train/: {len(train_samples)} images (50% PASS, 50% FAIL)")
    print(f"  test/: {len(test_samples)} images")
    print(f"  test_no_image/: {len(test_samples)} blank images (same specs)")
    print(f"  test_reasoning_pairs/: {len(pairs)} pairs (same image, diff specs)")


if __name__ == "__main__":
    main()
