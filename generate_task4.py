"""Task 4: Alignment offset — precision spatial measurement.

Two holes on a plate are nearly horizontally aligned. The model must
determine the vertical offset between their centers in mm using the
scale bar.

This is a Level 3 precision task: same visual elements as Tasks 1-2
(holes, plate, scale bar) but the measured quantity is much smaller
relative to the visible features. A 15mm hole is easy to see. A 2mm
vertical offset between two holes that are 40mm apart horizontally
is subtle.

Anti-shortcut measures:
  1. Offset independent of hole diameters
  2. Offset independent of horizontal separation
  3. Variable zoom breaks pixel-to-mm correlation
  4. Variable scale bar values
  5. Offset direction randomized (H2 above or below H1)
  6. Hole diameters vary independently of each other and offset

The formula is the same as Tasks 1-2:
    offset_mm = offset_pixels * (scale_bar_mm / scale_bar_pixels)

But the precision demand is higher because offsets are small (0.5-8mm).
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
OFFSET_RANGE: tuple[float, float] = (0.5, 8.0)
HOLE_DIAM_RANGE: tuple[float, float] = (5.0, 20.0)
HORIZ_SEP_RANGE: tuple[float, float] = (15.0, 50.0)
CANVAS_W_RANGE: tuple[int, int] = (500, 1000)
CANVAS_H_RANGE: tuple[int, int] = (400, 800)
DPI: int = 150

PLATE_COLOR: str = "#F0F0EA"
HOLE_FILL: str = "#333333"


def compute_ppm_range(
    h_sep_mm: float,
    offset_mm: float,
    d1_mm: float,
    d2_mm: float,
    sb_mm: int,
    canvas_w: int,
    canvas_h: int,
) -> tuple[float, float]:
    """Return feasible (min, max) pixels-per-mm."""
    max_horiz = h_sep_mm + max(d1_mm, d2_mm)
    max_vert = offset_mm + max(d1_mm, d2_mm)

    min_offset_px = 3
    min_sb_px = 30
    min_hole_px = 8

    ppm_min = max(
        min_offset_px / offset_mm,
        min_sb_px / sb_mm,
        min_hole_px / min(d1_mm, d2_mm),
    )

    max_frac = 0.50
    ppm_max = min(
        max_frac * canvas_w / max_horiz,
        max_frac * canvas_h / max_vert,
        max_frac * canvas_w / sb_mm,
    )

    return ppm_min, ppm_max


def generate_sample(rng: Generator, idx: int) -> dict[str, Any]:
    """Generate metadata for one image with two nearly-aligned holes."""
    offset_mm = float(rng.uniform(*OFFSET_RANGE))
    offset_mm = round(offset_mm, 2)
    offset_sign = rng.choice([-1, 1])

    d1_mm = float(rng.uniform(*HOLE_DIAM_RANGE))
    d2_mm = float(rng.uniform(*HOLE_DIAM_RANGE))
    h_sep_mm = float(rng.uniform(*HORIZ_SEP_RANGE))

    sb_mm = int(rng.choice(SCALE_BAR_VALUES))
    canvas_w = int(rng.integers(*CANVAS_W_RANGE))
    canvas_h = int(rng.integers(*CANVAS_H_RANGE))

    ppm_min, ppm_max = compute_ppm_range(
        h_sep_mm, offset_mm, d1_mm, d2_mm, sb_mm, canvas_w, canvas_h
    )

    if ppm_min >= ppm_max:
        canvas_w = int(canvas_w * 1.5)
        canvas_h = int(canvas_h * 1.5)
        ppm_min, ppm_max = compute_ppm_range(
            h_sep_mm, offset_mm, d1_mm, d2_mm, sb_mm, canvas_w, canvas_h
        )

    if ppm_min >= ppm_max:
        ppm = ppm_min + 0.5
    else:
        ppm = float(rng.uniform(ppm_min, ppm_max))

    h_sep_px = h_sep_mm * ppm
    offset_px = offset_mm * ppm * offset_sign
    d1_px = d1_mm * ppm
    d2_px = d2_mm * ppm
    sb_px = sb_mm * ppm

    plate_margin = 60
    plate_w = max(h_sep_px + max(d1_px, d2_px) + 40, 200)
    plate_h = max(abs(offset_px) + max(d1_px, d2_px) + 60, 150)
    plate_w = min(plate_w, canvas_w - 2 * plate_margin)
    plate_h = min(plate_h, canvas_h - 2 * plate_margin - 40)

    plate_x = (canvas_w - plate_w) / 2
    plate_y = 70.0

    center_y = plate_y + plate_h / 2
    h1_x = plate_x + plate_w * 0.3
    h1_y = center_y
    h2_x = plate_x + plate_w * 0.7
    h2_y = center_y + offset_px

    actual_offset_px = abs(h2_y - h1_y)
    actual_offset_mm = actual_offset_px / ppm

    sb_x = plate_x
    sb_y = plate_y - 35

    return {
        "idx": idx,
        "offset_mm": round(float(actual_offset_mm), 2),
        "offset_direction": "above" if offset_sign > 0 else "below",
        "h_sep_mm": round(h_sep_mm, 2),
        "d1_mm": round(d1_mm, 2),
        "d2_mm": round(d2_mm, 2),
        "scale_bar_mm": sb_mm,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "ppm": round(float(ppm), 4),
        "offset_px": round(float(actual_offset_px), 1),
        "h_sep_px": round(float(h_sep_px), 1),
        "d1_px": round(float(d1_px), 1),
        "d2_px": round(float(d2_px), 1),
        "sb_px": round(float(sb_px), 1),
        "h1_x": round(float(h1_x), 1),
        "h1_y": round(float(h1_y), 1),
        "h2_x": round(float(h2_x), 1),
        "h2_y": round(float(h2_y), 1),
        "plate_x": float(plate_x),
        "plate_y": float(plate_y),
        "plate_w_px": float(plate_w),
        "plate_h_px": float(plate_h),
        "sb_x": float(sb_x),
        "sb_y": float(sb_y),
    }


def render_image(sample: dict[str, Any], output_path: str) -> None:
    """Render two nearly-aligned holes on a plate with a scale bar."""
    cw, ch = sample["canvas_w"], sample["canvas_h"]
    fig, ax = plt.subplots(1, 1, figsize=(cw / DPI, ch / DPI), dpi=DPI)
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.axis("off")

    rect = mpl_patches.Rectangle(
        (sample["plate_x"], sample["plate_y"]),
        sample["plate_w_px"], sample["plate_h_px"],
        linewidth=1.5, edgecolor="black",
        facecolor=PLATE_COLOR, zorder=1,
    )
    ax.add_patch(rect)

    for label, cx, cy, d_px in [
        ("H1", sample["h1_x"], sample["h1_y"], sample["d1_px"]),
        ("H2", sample["h2_x"], sample["h2_y"], sample["d2_px"]),
    ]:
        circle = mpl_patches.Circle(
            (cx, cy), d_px / 2,
            linewidth=1.0, edgecolor="black",
            facecolor=HOLE_FILL, zorder=2,
        )
        ax.add_patch(circle)

        ax.text(cx, cy + d_px / 2 + 10, label,
                ha="center", va="bottom", fontsize=7,
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
    """Verify no single feature predicts offset_mm."""
    offsets = np.array([s["offset_mm"] for s in samples])
    offset_pxs = np.array([s["offset_px"] for s in samples])
    sb_mms = np.array([s["scale_bar_mm"] for s in samples])
    sb_pxs = np.array([s["sb_px"] for s in samples])
    d1s = np.array([s["d1_mm"] for s in samples])
    d2s = np.array([s["d2_mm"] for s in samples])
    h_seps = np.array([s["h_sep_mm"] for s in samples])
    ppms = np.array([s["ppm"] for s in samples])
    canvas_ws = np.array([s["canvas_w"] for s in samples])
    correct = offset_pxs * sb_mms / sb_pxs

    print("=== SHORTCUT VERIFICATION ===")
    print(f"N = {len(samples)}")
    print(f"Offset range: {offsets.min():.2f} - {offsets.max():.2f} mm")
    print(f"Offset mean: {offsets.mean():.2f} mm\n")

    features = {
        "offset_pixels": offset_pxs,
        "scale_bar_mm": sb_mms,
        "scale_bar_pixels": sb_pxs,
        "hole_d1_mm": d1s,
        "hole_d2_mm": d2s,
        "horiz_separation_mm": h_seps,
        "pixels_per_mm": ppms,
        "canvas_width": canvas_ws,
        "CORRECT (offset_px * sb_mm / sb_px)": correct,
    }

    print("Correlation with offset_mm:")
    for name, feat in features.items():
        r = np.corrcoef(offsets, feat)[0, 1]
        print(f"  {name:<45s} r = {r:+.3f}")
    print()

    from numpy.polynomial import polynomial as P

    mean_mae = np.mean(np.abs(offsets - offsets.mean()))
    c = P.polyfit(offset_pxs, offsets, 1)
    lr_mae = np.mean(np.abs(offsets - P.polyval(offset_pxs, c)))
    correct_mae = np.mean(np.abs(offsets - correct))

    print("Baseline MAEs:")
    print(f"  Always guess mean ({offsets.mean():.2f}mm):   {mean_mae:.2f} mm")
    print(f"  Linear regress on pixel offset:   {lr_mae:.2f} mm")
    print(f"  Correct formula (scale bar):      {correct_mae:.4f} mm\n")


def generate_matched_pairs(rng: Generator, n_pairs: int = 50) -> list[dict[str, Any]]:
    """Generate pairs with identical pixel layout but different scale bars."""
    pairs: list[dict[str, Any]] = []
    for i in range(n_pairs * 2):
        if len(pairs) >= n_pairs:
            break

        sb_pair = rng.choice(SCALE_BAR_VALUES, size=2, replace=False)
        sb_a, sb_b = int(sb_pair[0]), int(sb_pair[1])

        canvas_w = int(rng.integers(600, 900))
        canvas_h = int(rng.integers(500, 700))

        ppm_a = float(rng.uniform(3.0, 10.0))
        ppm_b = float(rng.uniform(3.0, 10.0))
        while abs(ppm_a - ppm_b) < 1.5:
            ppm_b = float(rng.uniform(3.0, 10.0))

        offset_px = float(rng.uniform(5, 40))
        offset_sign = rng.choice([-1, 1])
        h_sep_px = float(rng.uniform(60, 200))
        d1_px = float(rng.uniform(20, 60))
        d2_px = float(rng.uniform(20, 60))

        offset_a = offset_px / ppm_a
        offset_b = offset_px / ppm_b
        sb_px_a = sb_a * ppm_a
        sb_px_b = sb_b * ppm_b

        max_feature = max(h_sep_px + max(d1_px, d2_px), sb_px_a, sb_px_b)
        if max_feature > canvas_w * 0.55:
            continue

        plate_w = max(h_sep_px + max(d1_px, d2_px) + 40, 200)
        plate_h = max(offset_px + max(d1_px, d2_px) + 60, 150)
        plate_x = (canvas_w - plate_w) / 2
        plate_y = 70.0
        center_y = plate_y + plate_h / 2

        base = {
            "canvas_w": canvas_w, "canvas_h": canvas_h,
            "offset_px": round(offset_px, 1),
            "offset_direction": "above" if offset_sign > 0 else "below",
            "h1_x": round(plate_x + plate_w * 0.3, 1),
            "h1_y": round(center_y, 1),
            "h2_x": round(plate_x + plate_w * 0.7, 1),
            "h2_y": round(center_y + offset_px * offset_sign, 1),
            "d1_px": round(d1_px, 1), "d2_px": round(d2_px, 1),
            "h_sep_px": round(h_sep_px, 1),
            "plate_x": plate_x, "plate_y": plate_y,
            "plate_w_px": plate_w, "plate_h_px": plate_h,
            "sb_x": plate_x, "sb_y": plate_y - 35,
        }

        var_a = {**base, "idx": i, "offset_mm": round(offset_a, 2),
                 "scale_bar_mm": sb_a, "ppm": round(ppm_a, 4),
                 "sb_px": round(sb_px_a, 1),
                 "d1_mm": round(d1_px / ppm_a, 2), "d2_mm": round(d2_px / ppm_a, 2),
                 "h_sep_mm": round(h_sep_px / ppm_a, 2)}
        var_b = {**base, "idx": i, "offset_mm": round(offset_b, 2),
                 "scale_bar_mm": sb_b, "ppm": round(ppm_b, 4),
                 "sb_px": round(sb_px_b, 1),
                 "d1_mm": round(d1_px / ppm_b, 2), "d2_mm": round(d2_px / ppm_b, 2),
                 "h_sep_mm": round(h_sep_px / ppm_b, 2)}

        pairs.append({
            "pair_id": len(pairs),
            "offset_px": round(offset_px, 1),
            "a": var_a, "b": var_b,
            "offset_a": round(offset_a, 2), "offset_b": round(offset_b, 2),
            "sb_a": sb_a, "sb_b": sb_b,
        })

    return pairs


def verify_matched_pairs(pairs: list[dict[str, Any]]) -> None:
    """Print matched pair statistics."""
    diffs = [abs(p["offset_a"] - p["offset_b"]) for p in pairs]
    print("\n=== MATCHED PAIR DIAGNOSTIC ===")
    print(f"N pairs: {len(pairs)}")
    print(f"Offset difference per pair: {np.mean(diffs):.2f} mm avg "
          f"(range {np.min(diffs):.2f} - {np.max(diffs):.2f})")
    print("Same pixel layout, different scale bars -> different offsets")
    for p in pairs[:5]:
        print(f"  Pair {p['pair_id']}: offset={p['offset_px']}px, "
              f"SB {p['sb_a']}mm->{p['offset_a']}mm vs "
              f"SB {p['sb_b']}mm->{p['offset_b']}mm "
              f"(diff={abs(p['offset_a'] - p['offset_b']):.2f}mm)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 4: alignment offset")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--output-dir", type=str, default="dataset_task4")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train_samples = [generate_sample(rng, i) for i in range(args.n_train)]
    test_samples = [generate_sample(rng, i) for i in range(args.n_test)]

    print("=== TRAIN SET ===")
    verify_no_shortcuts(train_samples)
    print("=== TEST SET ===")
    verify_no_shortcuts(test_samples)

    if args.verify_only:
        return

    out = Path(args.output_dir)
    for split in ["train", "test", "test_matched"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    for split, samples in [("train", train_samples), ("test", test_samples)]:
        with open(out / split / "metadata.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        for i, s in enumerate(samples):
            render_image(s, str(out / split / f"image_{s['idx']:04d}.png"))
            if (i + 1) % 100 == 0:
                print(f"  {split}: {i + 1}/{len(samples)}")

    print("\nGenerating matched pair diagnostic set...")
    matched_pairs = generate_matched_pairs(rng, n_pairs=50)

    with open(out / "test_matched" / "metadata.jsonl", "w") as f:
        for pair in matched_pairs:
            f.write(json.dumps(pair) + "\n")

    for pair in matched_pairs:
        for variant in ["a", "b"]:
            render_image(
                pair[variant],
                str(out / "test_matched" / f"pair_{pair['pair_id']:03d}_{variant}.png"),
            )

    print(f"  Generated {len(matched_pairs)} matched pairs")
    verify_matched_pairs(matched_pairs)

    print(f"\nDone. Output in {out}/")
    print(f"  train/: {len(train_samples)} images")
    print(f"  test/: {len(test_samples)} images")
    print(f"  test_matched/: {len(matched_pairs)} pairs")


if __name__ == "__main__":
    main()
