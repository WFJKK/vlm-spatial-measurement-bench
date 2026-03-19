"""Task 2: Distance between two points — shortcut-proof synthetic dataset.

Generates technical drawings with two marked points (P1, P2) on a plate
and a labeled scale bar. The only valid path to the correct answer is:

    distance_mm = pixel_distance(P1, P2) × (scale_bar_mm / scale_bar_pixels)

Anti-shortcut measures (same principles as Task 1):
  1. Continuous uniform distances — no clustering
  2. Variable zoom independent of distance
  3. Random point angles — not always horizontal
  4. Variable plate size independent of distance
  5. Scale bar value independent of distance and zoom
  6. Variable canvas size

What makes this harder than Task 1 (diameter estimation):
  - Model must locate TWO features and compute the gap
  - Diagonal distances require Euclidean computation across patches
  - Points are small markers, not large circles — harder to locate
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
DISTANCE_RANGE: tuple[float, float] = (5.0, 40.0)
CANVAS_W_RANGE: tuple[int, int] = (500, 1000)
CANVAS_H_RANGE: tuple[int, int] = (400, 800)
DPI: int = 150

PLATE_COLOR: str = "#F0F0EA"
PLATE_EDGE_COLOR: str = "black"
POINT_COLOR: str = "red"
POINT_RADIUS_PX: float = 4.0


def compute_ppm_range(
    distance_mm: float,
    scale_bar_mm: int,
    canvas_w: int,
    canvas_h: int,
) -> tuple[float, float]:
    """Return feasible (min, max) pixels-per-mm so points and scale bar fit."""
    min_dist_px = 20
    min_sb_px = 30
    max_feature_frac = 0.50

    ppm_min = max(min_dist_px / distance_mm, min_sb_px / scale_bar_mm)
    ppm_max = max_feature_frac * min(canvas_w, canvas_h) / distance_mm
    ppm_max = min(ppm_max, max_feature_frac * canvas_w / scale_bar_mm)

    return ppm_min, ppm_max


def generate_sample(rng: Generator, idx: int) -> dict[str, Any]:
    """Generate metadata for one image with two randomly placed points."""
    distance_mm = rng.uniform(*DISTANCE_RANGE)
    scale_bar_mm = int(rng.choice(SCALE_BAR_VALUES))
    canvas_w = int(rng.integers(*CANVAS_W_RANGE))
    canvas_h = int(rng.integers(*CANVAS_H_RANGE))

    ppm_min, ppm_max = compute_ppm_range(distance_mm, scale_bar_mm, canvas_w, canvas_h)

    if ppm_min >= ppm_max:
        canvas_w = int(canvas_w * 1.5)
        canvas_h = int(canvas_h * 1.5)
        ppm_min, ppm_max = compute_ppm_range(distance_mm, scale_bar_mm, canvas_w, canvas_h)

    ppm = rng.uniform(ppm_min, ppm_max)
    dist_px = distance_mm * ppm
    sb_px = scale_bar_mm * ppm

    plate_margin = 60
    plate_w = max(dist_px * 1.5, 150)
    plate_h = max(dist_px * 1.5, 120)
    plate_w = min(plate_w, canvas_w - 2 * plate_margin)
    plate_h = min(plate_h, canvas_h - 2 * plate_margin - 40)

    plate_x = (canvas_w - plate_w) / 2
    plate_y = 70.0

    angle = rng.uniform(0, 2 * np.pi)
    mid_x = plate_x + plate_w / 2
    mid_y = plate_y + plate_h / 2

    half_dist = dist_px / 2
    dx = half_dist * np.cos(angle)
    dy = half_dist * np.sin(angle)

    p1_x = mid_x - dx
    p1_y = mid_y - dy
    p2_x = mid_x + dx
    p2_y = mid_y + dy

    point_margin = 15
    p1_x = np.clip(p1_x, plate_x + point_margin, plate_x + plate_w - point_margin)
    p1_y = np.clip(p1_y, plate_y + point_margin, plate_y + plate_h - point_margin)
    p2_x = np.clip(p2_x, plate_x + point_margin, plate_x + plate_w - point_margin)
    p2_y = np.clip(p2_y, plate_y + point_margin, plate_y + plate_h - point_margin)

    actual_dist_px = np.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)
    actual_dist_mm = actual_dist_px / ppm

    sb_x = plate_x
    sb_y = plate_y - 35

    return {
        "idx": idx,
        "distance_mm": round(float(actual_dist_mm), 2),
        "scale_bar_mm": scale_bar_mm,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "ppm": round(float(ppm), 4),
        "dist_px": round(float(actual_dist_px), 1),
        "sb_px": round(float(sb_px), 1),
        "p1_x": round(float(p1_x), 1),
        "p1_y": round(float(p1_y), 1),
        "p2_x": round(float(p2_x), 1),
        "p2_y": round(float(p2_y), 1),
        "angle_rad": round(float(angle), 3),
        "plate_x": float(plate_x),
        "plate_y": float(plate_y),
        "plate_w_px": float(plate_w),
        "plate_h_px": float(plate_h),
        "sb_x": float(sb_x),
        "sb_y": float(sb_y),
    }


def render_image(sample: dict[str, Any], output_path: str) -> None:
    """Render a technical drawing with two labeled points and a scale bar."""
    cw, ch = sample["canvas_w"], sample["canvas_h"]
    fig, ax = plt.subplots(1, 1, figsize=(cw / DPI, ch / DPI), dpi=DPI)
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.axis("off")

    rect = mpl_patches.Rectangle(
        (sample["plate_x"], sample["plate_y"]),
        sample["plate_w_px"], sample["plate_h_px"],
        linewidth=1.5, edgecolor=PLATE_EDGE_COLOR,
        facecolor=PLATE_COLOR, zorder=1,
    )
    ax.add_patch(rect)

    for label, px, py in [("P1", sample["p1_x"], sample["p1_y"]),
                           ("P2", sample["p2_x"], sample["p2_y"])]:
        circle = mpl_patches.Circle(
            (px, py), POINT_RADIUS_PX,
            linewidth=1.0, edgecolor="black",
            facecolor=POINT_COLOR, zorder=3,
        )
        ax.add_patch(circle)

        cross_len = POINT_RADIUS_PX * 2.5
        ax.plot([px - cross_len, px + cross_len], [py, py],
                color="#666666", linewidth=0.6, zorder=2)
        ax.plot([px, px], [py - cross_len, py + cross_len],
                color="#666666", linewidth=0.6, zorder=2)

        ax.text(px, py + POINT_RADIUS_PX + 8, label,
                ha="center", va="bottom", fontsize=7,
                fontweight="bold", zorder=4)

    sb_x, sb_y, sb_len = sample["sb_x"], sample["sb_y"], sample["sb_px"]
    ax.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color="black", linewidth=2, zorder=5)
    tick_h = 5
    ax.plot([sb_x, sb_x], [sb_y - tick_h, sb_y + tick_h], color="black", linewidth=1.5, zorder=5)
    ax.plot([sb_x + sb_len, sb_x + sb_len], [sb_y - tick_h, sb_y + tick_h], color="black", linewidth=1.5, zorder=5)
    ax.text(sb_x + sb_len / 2, sb_y - 10, f"{sample['scale_bar_mm']} mm",
            ha="center", va="top", fontsize=7, zorder=5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def verify_no_shortcuts(samples: list[dict[str, Any]]) -> None:
    """Verify that no single feature predicts distance_mm."""
    dists = np.array([s["distance_mm"] for s in samples])
    dist_pxs = np.array([s["dist_px"] for s in samples])
    sb_mms = np.array([s["scale_bar_mm"] for s in samples])
    sb_pxs = np.array([s["sb_px"] for s in samples])
    canvas_ws = np.array([s["canvas_w"] for s in samples])
    ppms = np.array([s["ppm"] for s in samples])
    angles = np.array([s["angle_rad"] for s in samples])
    correct = dist_pxs * sb_mms / sb_pxs

    print("=== SHORTCUT VERIFICATION ===")
    print(f"N = {len(samples)}")
    print(f"Distance range: {dists.min():.1f} - {dists.max():.1f} mm")
    print(f"Distance mean: {dists.mean():.1f} mm\n")

    features = {
        "dist_pixels": dist_pxs,
        "scale_bar_mm": sb_mms,
        "scale_bar_pixels": sb_pxs,
        "canvas_width": canvas_ws,
        "pixels_per_mm": ppms,
        "angle": angles,
        "dist_px / canvas_w": dist_pxs / canvas_ws,
        "CORRECT (dist_px * sb_mm / sb_px)": correct,
    }

    print("Correlation with distance_mm:")
    for name, feat in features.items():
        r = np.corrcoef(dists, feat)[0, 1]
        print(f"  {name:<45s} r = {r:+.3f}")
    print()

    from numpy.polynomial import polynomial as P

    mean_mae = np.mean(np.abs(dists - dists.mean()))
    c = P.polyfit(dist_pxs, dists, 1)
    lr_mae = np.mean(np.abs(dists - P.polyval(dist_pxs, c)))
    correct_mae = np.mean(np.abs(dists - correct))

    print("Baseline MAEs:")
    print(f"  Always guess mean ({dists.mean():.1f}mm):     {mean_mae:.2f} mm")
    print(f"  Linear regress on pixel distance:  {lr_mae:.2f} mm")
    print(f"  Correct formula (scale bar):       {correct_mae:.4f} mm\n")


def generate_matched_pairs(rng: Generator, n_pairs: int = 50) -> list[dict[str, Any]]:
    """Generate image pairs with identical pixel point positions but different scale bars."""
    pairs: list[dict[str, Any]] = []
    for i in range(n_pairs):
        sb_pair = rng.choice(SCALE_BAR_VALUES, size=2, replace=False)
        sb_a, sb_b = int(sb_pair[0]), int(sb_pair[1])

        canvas_w = int(rng.integers(600, 900))
        canvas_h = int(rng.integers(500, 700))

        angle = rng.uniform(0, 2 * np.pi)
        dist_px = float(rng.uniform(30, 200))

        ppm_a = float(rng.uniform(3.0, 12.0))
        ppm_b = float(rng.uniform(3.0, 12.0))
        while abs(ppm_a - ppm_b) < 1.5:
            ppm_b = float(rng.uniform(3.0, 12.0))

        dist_a = dist_px / ppm_a
        dist_b = dist_px / ppm_b
        sb_px_a = sb_a * ppm_a
        sb_px_b = sb_b * ppm_b

        max_feature = max(dist_px, sb_px_a, sb_px_b)
        if max_feature > canvas_w * 0.55 or dist_px > canvas_h * 0.4:
            continue

        plate_w = max(dist_px * 1.5, 150)
        plate_h = max(dist_px * 1.5, 120)
        plate_x = (canvas_w - plate_w) / 2
        plate_y = 70.0
        mid_x = plate_x + plate_w / 2
        mid_y = plate_y + plate_h / 2

        half_dist = dist_px / 2
        dx = half_dist * np.cos(angle)
        dy = half_dist * np.sin(angle)

        base = {
            "canvas_w": canvas_w, "canvas_h": canvas_h,
            "dist_px": round(dist_px, 1),
            "p1_x": round(mid_x - dx, 1), "p1_y": round(mid_y - dy, 1),
            "p2_x": round(mid_x + dx, 1), "p2_y": round(mid_y + dy, 1),
            "angle_rad": round(float(angle), 3),
            "plate_x": plate_x, "plate_y": plate_y,
            "plate_w_px": plate_w, "plate_h_px": plate_h,
            "sb_x": plate_x, "sb_y": plate_y - 35,
        }

        var_a = {**base, "idx": i, "distance_mm": round(dist_a, 2), "scale_bar_mm": sb_a, "ppm": round(ppm_a, 4), "sb_px": round(sb_px_a, 1)}
        var_b = {**base, "idx": i, "distance_mm": round(dist_b, 2), "scale_bar_mm": sb_b, "ppm": round(ppm_b, 4), "sb_px": round(sb_px_b, 1)}

        pairs.append({
            "pair_id": i, "dist_px": round(dist_px, 1),
            "a": var_a, "b": var_b,
            "dist_a": round(dist_a, 2), "dist_b": round(dist_b, 2),
            "sb_a": sb_a, "sb_b": sb_b,
        })

    return pairs


def verify_matched_pairs(pairs: list[dict[str, Any]]) -> None:
    """Print matched pair statistics."""
    diffs = [abs(p["dist_a"] - p["dist_b"]) for p in pairs]
    print("\n=== MATCHED PAIR DIAGNOSTIC ===")
    print(f"N pairs: {len(pairs)}")
    print(f"Ground truth difference per pair: {np.mean(diffs):.1f} mm avg "
          f"(range {np.min(diffs):.1f} - {np.max(diffs):.1f})")
    print("Same pixel positions, different scale bars -> different correct distances")
    print("If model gives same answer for both: NOT using scale bar")
    print("If model answers diverge correctly: USING scale bar\n")
    for p in pairs[:5]:
        print(f"  Pair {p['pair_id']}: dist={p['dist_px']}px, "
              f"SB {p['sb_a']}mm->{p['dist_a']}mm vs SB {p['sb_b']}mm->{p['dist_b']}mm "
              f"(diff={abs(p['dist_a'] - p['dist_b']):.1f}mm)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 2: distance between two points")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-dir", type=str, default="dataset_task2")
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
            render_image(pair[variant], str(out / "test_matched" / f"pair_{pair['pair_id']:03d}_{variant}.png"))

    print(f"  Generated {len(matched_pairs)} matched pairs")
    verify_matched_pairs(matched_pairs)

    print(f"\nDone. Output in {out}/")
    print(f"  train/: {len(train_samples)} images")
    print(f"  test/: {len(test_samples)} images")
    print(f"  test_matched/: {len(matched_pairs)} pairs ({len(matched_pairs) * 2} images)")


if __name__ == "__main__":
    main()
