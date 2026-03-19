"""Task 3: Gauge reading — shortcut-proof synthetic dataset.

Generates images of circular gauges with numbered tick marks and a
pointer. The model must determine the reading by interpolating the
pointer's angular position between scale markings.

This tests a fundamentally different spatial mechanism than Tasks 1-2:
angular interpolation instead of linear measurement with a scale bar.

Anti-shortcut measures:
  1. Variable min/max range (not always 0-100)
  2. Variable number of major ticks
  3. Continuous pointer positions — no alignment with ticks
  4. Variable gauge size and position on canvas
  5. Variable arc span (not always full circle)
  6. Pointer angle does NOT correlate with reading when range varies

Connects to MeasureBench: their core task is instrument reading.
If the decoding bottleneck exists here too, it's truly universal.
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

CANVAS_W_RANGE: tuple[int, int] = (400, 800)
CANVAS_H_RANGE: tuple[int, int] = (400, 800)
DPI: int = 150

RANGE_CONFIGS: list[tuple[float, float]] = [
    (0, 100), (0, 50), (0, 200), (0, 10),
    (-20, 60), (0, 1000), (10, 90), (0, 500),
    (-50, 50), (0, 150), (20, 120), (0, 30),
]
MAJOR_TICK_COUNTS: list[int] = [5, 6, 8, 10, 11]
ARC_SPANS: list[float] = [180.0, 210.0, 240.0, 270.0, 300.0]


def generate_sample(rng: Generator, idx: int) -> dict[str, Any]:
    """Generate metadata for one gauge image."""
    canvas_w = int(rng.integers(*CANVAS_W_RANGE))
    canvas_h = int(rng.integers(*CANVAS_H_RANGE))

    range_min, range_max = RANGE_CONFIGS[rng.integers(0, len(RANGE_CONFIGS))]
    n_major = int(rng.choice(MAJOR_TICK_COUNTS))
    arc_span = float(rng.choice(ARC_SPANS))

    reading = float(rng.uniform(range_min, range_max))
    reading = round(reading, 1)

    gauge_radius_frac = rng.uniform(0.25, 0.40)
    gauge_radius = gauge_radius_frac * min(canvas_w, canvas_h)

    cx = canvas_w / 2 + rng.uniform(-canvas_w * 0.1, canvas_w * 0.1)
    cy = canvas_h / 2 + rng.uniform(-canvas_h * 0.1, canvas_h * 0.1)

    cx = float(np.clip(cx, gauge_radius + 30, canvas_w - gauge_radius - 30))
    cy = float(np.clip(cy, gauge_radius + 30, canvas_h - gauge_radius - 30))

    arc_start = 180 + (360 - arc_span) / 2
    arc_end = arc_start + arc_span

    frac = (reading - range_min) / (range_max - range_min)
    pointer_angle_deg = arc_start + frac * arc_span

    return {
        "idx": idx,
        "reading": reading,
        "range_min": range_min,
        "range_max": range_max,
        "n_major_ticks": n_major,
        "arc_span_deg": arc_span,
        "arc_start_deg": round(arc_start, 1),
        "arc_end_deg": round(arc_end, 1),
        "gauge_radius": round(gauge_radius, 1),
        "cx": round(cx, 1),
        "cy": round(cy, 1),
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "pointer_angle_deg": round(pointer_angle_deg, 2),
        "frac": round(frac, 4),
    }


def render_image(sample: dict[str, Any], output_path: str) -> None:
    """Render a gauge with numbered ticks and a pointer."""
    cw, ch = sample["canvas_w"], sample["canvas_h"]
    fig, ax = plt.subplots(1, 1, figsize=(cw / DPI, ch / DPI), dpi=DPI)
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.axis("off")

    cx, cy = sample["cx"], sample["cy"]
    r = sample["gauge_radius"]
    arc_start = sample["arc_start_deg"]
    arc_span = sample["arc_span_deg"]
    range_min = sample["range_min"]
    range_max = sample["range_max"]
    n_major = sample["n_major_ticks"]

    outer_ring = mpl_patches.Arc(
        (cx, cy), 2 * r, 2 * r,
        angle=0, theta1=arc_start, theta2=arc_start + arc_span,
        linewidth=2.5, edgecolor="black", zorder=2,
    )
    ax.add_patch(outer_ring)

    inner_ring = mpl_patches.Circle(
        (cx, cy), r * 0.08,
        facecolor="#333333", edgecolor="black", linewidth=1, zorder=5,
    )
    ax.add_patch(inner_ring)

    bg_circle = mpl_patches.Circle(
        (cx, cy), r * 1.05,
        facecolor="#F8F8F0", edgecolor="#CCCCCC", linewidth=1.5, zorder=1,
    )
    ax.add_patch(bg_circle)

    for i in range(n_major + 1):
        frac_tick = i / n_major
        angle_deg = arc_start + frac_tick * arc_span
        angle_rad = np.radians(angle_deg)

        x_outer = cx + r * np.cos(angle_rad)
        y_outer = cy + r * np.sin(angle_rad)
        x_inner = cx + r * 0.85 * np.cos(angle_rad)
        y_inner = cy + r * 0.85 * np.sin(angle_rad)

        ax.plot([x_inner, x_outer], [y_inner, y_outer],
                color="black", linewidth=1.5, zorder=3)

        value = range_min + frac_tick * (range_max - range_min)
        if value == int(value):
            label = str(int(value))
        else:
            label = f"{value:.1f}"

        x_label = cx + r * 0.72 * np.cos(angle_rad)
        y_label = cy + r * 0.72 * np.sin(angle_rad)

        fontsize = max(5, min(8, r * 0.06))
        ax.text(x_label, y_label, label,
                ha="center", va="center", fontsize=fontsize, zorder=4)

    n_minor_per_major = 4
    for i in range(n_major * n_minor_per_major + 1):
        frac_tick = i / (n_major * n_minor_per_major)
        if i % n_minor_per_major == 0:
            continue
        angle_deg = arc_start + frac_tick * arc_span
        angle_rad = np.radians(angle_deg)

        x_outer = cx + r * np.cos(angle_rad)
        y_outer = cy + r * np.sin(angle_rad)
        x_inner = cx + r * 0.92 * np.cos(angle_rad)
        y_inner = cy + r * 0.92 * np.sin(angle_rad)

        ax.plot([x_inner, x_outer], [y_inner, y_outer],
                color="#666666", linewidth=0.7, zorder=3)

    pointer_angle_rad = np.radians(sample["pointer_angle_deg"])
    pointer_len = r * 0.78
    px = cx + pointer_len * np.cos(pointer_angle_rad)
    py = cy + pointer_len * np.sin(pointer_angle_rad)

    ax.annotate("", xy=(px, py), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="red",
                                lw=2, mutation_scale=12),
                zorder=6)

    unit_y = cy - r * 1.2
    if unit_y < 15:
        unit_y = cy + r * 1.15
    ax.text(cx, unit_y, "Reading = ?",
            ha="center", va="center", fontsize=7,
            style="italic", color="#666666", zorder=4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)


def verify_no_shortcuts(samples: list[dict[str, Any]]) -> None:
    """Verify no single feature predicts the reading."""
    readings = np.array([s["reading"] for s in samples])
    angles = np.array([s["pointer_angle_deg"] for s in samples])
    fracs = np.array([s["frac"] for s in samples])
    radii = np.array([s["gauge_radius"] for s in samples])
    canvas_ws = np.array([s["canvas_w"] for s in samples])
    range_mins = np.array([s["range_min"] for s in samples])
    range_maxs = np.array([s["range_max"] for s in samples])
    range_spans = range_maxs - range_mins

    print("=== SHORTCUT VERIFICATION ===")
    print(f"N = {len(samples)}")
    print(f"Reading range: {readings.min():.1f} - {readings.max():.1f}")
    print(f"Reading mean: {readings.mean():.1f}\n")

    features = {
        "pointer_angle_deg": angles,
        "fractional_position": fracs,
        "gauge_radius": radii,
        "canvas_width": canvas_ws,
        "range_min": range_mins,
        "range_max": range_maxs,
        "range_span": range_spans,
    }

    print("Correlation with reading:")
    for name, feat in features.items():
        r = np.corrcoef(readings, feat)[0, 1]
        print(f"  {name:<35s} r = {r:+.3f}")

    correct = range_mins + fracs * range_spans
    correct_r = np.corrcoef(readings, correct)[0, 1]
    print(f"  {'CORRECT (min + frac * span)':<35s} r = {correct_r:+.3f}")
    print()

    mean_mae = np.mean(np.abs(readings - readings.mean()))
    from numpy.polynomial import polynomial as P
    c = P.polyfit(angles, readings, 1)
    angle_pred = P.polyval(angles, c)
    angle_mae = np.mean(np.abs(readings - angle_pred))
    correct_mae = np.mean(np.abs(readings - correct))

    print("Baseline MAEs:")
    print(f"  Always guess mean ({readings.mean():.1f}):    {mean_mae:.2f}")
    print(f"  Linear regress on angle:         {angle_mae:.2f}")
    print(f"  Correct formula:                 {correct_mae:.4f}\n")


def generate_matched_pairs(rng: Generator, n_pairs: int = 50) -> list[dict[str, Any]]:
    """Generate pairs with same pointer angle but different scales.

    Same visual pointer position, different range_min/range_max,
    so the reading differs. Tests whether the model reads the scale
    markings, not just the pointer angle.
    """
    pairs: list[dict[str, Any]] = []
    for i in range(n_pairs * 2):
        if len(pairs) >= n_pairs:
            break

        canvas_w = int(rng.integers(450, 750))
        canvas_h = int(rng.integers(450, 750))
        n_major = int(rng.choice(MAJOR_TICK_COUNTS))
        arc_span = float(rng.choice(ARC_SPANS))
        arc_start = 180 + (360 - arc_span) / 2

        gauge_radius = float(rng.uniform(0.28, 0.38)) * min(canvas_w, canvas_h)
        cx = float(canvas_w / 2)
        cy = float(canvas_h / 2)

        frac = float(rng.uniform(0.1, 0.9))
        pointer_angle_deg = arc_start + frac * arc_span

        idx_a = rng.integers(0, len(RANGE_CONFIGS))
        idx_b = rng.integers(0, len(RANGE_CONFIGS))
        while idx_b == idx_a:
            idx_b = rng.integers(0, len(RANGE_CONFIGS))

        range_min_a, range_max_a = RANGE_CONFIGS[idx_a]
        range_min_b, range_max_b = RANGE_CONFIGS[idx_b]

        reading_a = round(range_min_a + frac * (range_max_a - range_min_a), 1)
        reading_b = round(range_min_b + frac * (range_max_b - range_min_b), 1)

        if abs(reading_a - reading_b) < 1.0:
            continue

        base = {
            "canvas_w": canvas_w, "canvas_h": canvas_h,
            "n_major_ticks": n_major,
            "arc_span_deg": arc_span,
            "arc_start_deg": round(arc_start, 1),
            "arc_end_deg": round(arc_start + arc_span, 1),
            "gauge_radius": round(gauge_radius, 1),
            "cx": round(cx, 1), "cy": round(cy, 1),
            "pointer_angle_deg": round(pointer_angle_deg, 2),
            "frac": round(frac, 4),
        }

        var_a = {**base, "idx": i, "reading": reading_a, "range_min": range_min_a, "range_max": range_max_a}
        var_b = {**base, "idx": i, "reading": reading_b, "range_min": range_min_b, "range_max": range_max_b}

        pairs.append({
            "pair_id": len(pairs),
            "a": var_a, "b": var_b,
            "reading_a": reading_a, "reading_b": reading_b,
            "pointer_angle_deg": round(pointer_angle_deg, 2),
            "frac": round(frac, 4),
        })

    return pairs


def verify_matched_pairs(pairs: list[dict[str, Any]]) -> None:
    """Print matched pair statistics."""
    diffs = [abs(p["reading_a"] - p["reading_b"]) for p in pairs]
    print("\n=== MATCHED PAIR DIAGNOSTIC ===")
    print(f"N pairs: {len(pairs)}")
    print(f"Reading difference per pair: {np.mean(diffs):.1f} avg "
          f"(range {np.min(diffs):.1f} - {np.max(diffs):.1f})")
    print("Same pointer angle, different scales -> different correct readings")
    print("If model gives same answer: reading angle only, NOT the scale")
    print("If answers diverge correctly: reading both angle AND scale\n")
    for p in pairs[:5]:
        print(f"  Pair {p['pair_id']}: angle={p['pointer_angle_deg']:.0f}°, "
              f"reading {p['reading_a']} vs {p['reading_b']} "
              f"(diff={abs(p['reading_a'] - p['reading_b']):.1f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Task 3: gauge reading")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--output-dir", type=str, default="dataset_task3")
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
