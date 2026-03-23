"""Task 7: Realistic compliance plates — multi-feature inspection.

Generates plates DESIGNED to meet a spec, with 1-2 intentional violations
introduced as manufacturing defects. The model must find ALL violations.

This tests whether the model can inspect multiple features and catch
subtle non-compliance. More realistic than random holes.

Output format: list of violations as "H2 diameter: 16.3mm (spec max 15mm)"
or "PASS" if no violations.

Quick experiment: baseline eval only, ~20 min. No training needed to see
whether this is a useful task direction.

Usage:
    python3 task7_compliance.py --generate
    python3 task7_compliance.py --baseline
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

MODEL_ID: str = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_DIR: str = "dataset_task7"
SCALE_BAR_VALUES: list[int] = [10, 15, 20, 25, 30]


def generate_dataset(n_test: int = 100, seed: int = 48) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpl_patches

    rng = np.random.default_rng(seed)
    out = Path(DATASET_DIR) / "test"
    out.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []

    for idx in range(n_test):
        # Design a spec
        n_holes = int(rng.integers(3, 7))
        diam_min = round(float(rng.uniform(8, 12)), 1)
        diam_max = round(diam_min + float(rng.uniform(3, 8)), 1)
        min_spacing = round(float(rng.uniform(15, 30)), 1)

        spec = {
            "diam_min": diam_min,
            "diam_max": diam_max,
            "min_spacing": min_spacing,
            "spec_text": (
                f"All hole diameters must be between {diam_min} mm and {diam_max} mm. "
                f"Minimum spacing between any two holes must be {min_spacing} mm."
            ),
        }

        # Generate holes that mostly comply
        sb_mm = int(rng.choice(SCALE_BAR_VALUES))
        canvas_w = int(rng.integers(600, 1000))
        canvas_h = int(rng.integers(500, 800))

        nom_diam = (diam_min + diam_max) / 2
        ppm_min = max(12 / diam_min, 30 / sb_mm)
        ppm_max = min(0.3 * canvas_w / diam_max, 0.3 * canvas_w / sb_mm)
        if ppm_min >= ppm_max:
            canvas_w = int(canvas_w * 1.5)
            canvas_h = int(canvas_h * 1.5)
            ppm_max = min(0.3 * canvas_w / diam_max, 0.3 * canvas_w / sb_mm)
        ppm = float(rng.uniform(ppm_min, min(ppm_max, ppm_min * 3)))
        sb_px = sb_mm * ppm

        plate_margin = 50
        plate_w = canvas_w - 2 * plate_margin
        plate_h = canvas_h - 2 * plate_margin - 40
        plate_x = float(plate_margin)
        plate_y = 70.0

        # Place holes with compliant diameters and spacing
        holes: list[dict[str, Any]] = []
        for i in range(n_holes):
            d_mm = round(float(rng.uniform(diam_min + 0.5, diam_max - 0.5)), 2)
            d_px = d_mm * ppm

            placed = False
            for _ in range(100):
                cx = plate_x + rng.uniform(d_px / 2 + 10, plate_w - d_px / 2 - 10)
                cy = plate_y + rng.uniform(d_px / 2 + 10, plate_h - d_px / 2 - 10)

                ok = True
                for h in holes:
                    dist_px = np.sqrt((cx - h["cx"]) ** 2 + (cy - h["cy"]) ** 2)
                    dist_mm = dist_px / ppm
                    if dist_mm < min_spacing + 2:
                        ok = False
                        break
                if ok:
                    placed = True
                    break

            if not placed:
                cx = plate_x + plate_w / 2
                cy = plate_y + plate_h / 2

            holes.append({
                "label": f"H{i + 1}",
                "d_mm": d_mm,
                "d_px": round(d_mm * ppm, 1),
                "cx": round(float(cx), 1),
                "cy": round(float(cy), 1),
                "compliant_diam": True,
            })

        # Introduce 1-2 violations
        n_violations = int(rng.choice([1, 2]))
        violation_type = rng.choice(["diameter", "spacing", "both"])
        violations: list[dict[str, str]] = []

        if violation_type in ["diameter", "both"]:
            # Make one hole too big or too small
            vic = int(rng.integers(0, len(holes)))
            if rng.random() > 0.5:
                new_d = round(diam_max + float(rng.uniform(0.5, 3.0)), 2)
                violations.append({
                    "type": "diameter_over",
                    "hole": holes[vic]["label"],
                    "measured": new_d,
                    "limit": diam_max,
                    "violation_mm": round(new_d - diam_max, 2),
                })
            else:
                new_d = round(diam_min - float(rng.uniform(0.5, 3.0)), 2)
                new_d = max(1.0, new_d)
                violations.append({
                    "type": "diameter_under",
                    "hole": holes[vic]["label"],
                    "measured": new_d,
                    "limit": diam_min,
                    "violation_mm": round(diam_min - new_d, 2),
                })
            holes[vic]["d_mm"] = new_d
            holes[vic]["d_px"] = round(new_d * ppm, 1)
            holes[vic]["compliant_diam"] = False

        if violation_type in ["spacing", "both"] and len(holes) >= 2:
            # Move two holes closer together
            h1_idx = int(rng.integers(0, len(holes)))
            h2_idx = (h1_idx + 1) % len(holes)
            target_spacing = min_spacing - float(rng.uniform(1.0, 5.0))
            target_spacing = max(3.0, target_spacing)
            target_px = target_spacing * ppm

            angle = float(rng.uniform(0, 2 * np.pi))
            mid_cx = (holes[h1_idx]["cx"] + holes[h2_idx]["cx"]) / 2
            mid_cy = (holes[h1_idx]["cy"] + holes[h2_idx]["cy"]) / 2
            holes[h1_idx]["cx"] = round(mid_cx - target_px / 2 * np.cos(angle), 1)
            holes[h1_idx]["cy"] = round(mid_cy - target_px / 2 * np.sin(angle), 1)
            holes[h2_idx]["cx"] = round(mid_cx + target_px / 2 * np.cos(angle), 1)
            holes[h2_idx]["cy"] = round(mid_cy + target_px / 2 * np.sin(angle), 1)

            violations.append({
                "type": "spacing",
                "holes": f"{holes[h1_idx]['label']}-{holes[h2_idx]['label']}",
                "measured": round(target_spacing, 2),
                "limit": min_spacing,
                "violation_mm": round(min_spacing - target_spacing, 2),
            })

        # If no violations were added (edge case), force a diameter violation
        if not violations:
            vic = 0
            new_d = round(diam_max + 1.5, 2)
            holes[vic]["d_mm"] = new_d
            holes[vic]["d_px"] = round(new_d * ppm, 1)
            violations.append({
                "type": "diameter_over",
                "hole": "H1",
                "measured": new_d,
                "limit": diam_max,
                "violation_mm": round(new_d - diam_max, 2),
            })

        total_violation = round(sum(v["violation_mm"] for v in violations), 2)

        sample = {
            "idx": idx,
            "n_holes": n_holes,
            "holes": holes,
            "spec": spec,
            "violations": violations,
            "n_violations": len(violations),
            "total_violation_mm": total_violation,
            "scale_bar_mm": sb_mm,
            "sb_px": round(sb_px, 1),
            "ppm": round(ppm, 4),
            "canvas_w": canvas_w,
            "canvas_h": canvas_h,
            "plate_x": plate_x,
            "plate_y": plate_y,
            "plate_w": plate_w,
            "plate_h": plate_h,
        }
        samples.append(sample)

        # Render
        cw, ch = canvas_w, canvas_h
        fig, ax = plt.subplots(1, 1, figsize=(cw / 150, ch / 150), dpi=150)
        ax.set_xlim(0, cw)
        ax.set_ylim(0, ch)
        ax.set_aspect("equal")
        ax.axis("off")

        rect = mpl_patches.Rectangle(
            (plate_x, plate_y), plate_w, plate_h,
            linewidth=1.5, edgecolor="black", facecolor="#F0F0EA", zorder=1,
        )
        ax.add_patch(rect)

        for h in holes:
            circle = mpl_patches.Circle(
                (h["cx"], h["cy"]), h["d_px"] / 2,
                linewidth=1.0, edgecolor="black", facecolor="#333333", zorder=2,
            )
            ax.add_patch(circle)
            ax.text(h["cx"], h["cy"] + h["d_px"] / 2 + 10, h["label"],
                    ha="center", va="bottom", fontsize=6, fontweight="bold", zorder=3)

        sb_x = plate_x
        sb_y = plate_y - 35
        ax.plot([sb_x, sb_x + sb_px], [sb_y, sb_y], color="black", linewidth=2, zorder=5)
        ax.plot([sb_x, sb_x], [sb_y - 5, sb_y + 5], color="black", linewidth=1.5, zorder=5)
        ax.plot([sb_x + sb_px, sb_x + sb_px], [sb_y - 5, sb_y + 5], color="black", linewidth=1.5, zorder=5)
        ax.text(sb_x + sb_px / 2, sb_y - 10, f"{sb_mm} mm",
                ha="center", va="top", fontsize=7, zorder=5)

        fig.savefig(str(out / f"image_{idx:04d}.png"), dpi=150,
                    bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)

        if (idx + 1) % 25 == 0:
            print(f"  {idx + 1}/{n_test}")

    with open(out / "metadata.jsonl", "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    diam_v = sum(1 for s in samples for v in s["violations"] if "diameter" in v["type"])
    space_v = sum(1 for s in samples for v in s["violations"] if v["type"] == "spacing")
    totals = [s["total_violation_mm"] for s in samples]
    print(f"\nGenerated {n_test} compliance plates:")
    print(f"  Diameter violations: {diam_v}")
    print(f"  Spacing violations: {space_v}")
    print(f"  Total violation range: {min(totals):.1f} - {max(totals):.1f} mm")
    print(f"  Mean: {np.mean(totals):.1f} mm")


def run_baseline() -> None:
    import re
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("\n=== Task 7 Baseline: Realistic Compliance ===\n")

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()

    with open(Path(DATASET_DIR) / "test" / "metadata.jsonl") as f:
        samples = [json.loads(l) for l in f]

    # Test 1: Can it find violations? (binary: any violation found?)
    # Test 2: Total violation amount (continuous)
    # Test 3: How many measurement calls would it need?

    prompt_binary = (
        "You are inspecting a technical drawing for compliance. {spec} "
        "Use the scale bar to measure all holes. "
        "Respond with PASS if all features comply, or FAIL if any violation exists."
    )

    prompt_total = (
        "You are inspecting a technical drawing for compliance. {spec} "
        "Use the scale bar to measure all holes and check all rules. "
        "Respond with ONLY the total amount of non-compliance in mm as a number. "
        "If everything passes, respond with 0."
    )

    def infer(model, processor, path, prompt):
        image = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    # Binary test
    print("--- Binary Detection (PASS/FAIL) ---")
    correct = 0
    total = 0
    for i, s in enumerate(samples):
        path = str(Path(DATASET_DIR) / "test" / f"image_{s['idx']:04d}.png")
        prompt = prompt_binary.format(spec=s["spec"]["spec_text"])
        raw = infer(model, processor, path, prompt)
        pred = "FAIL" if "FAIL" in raw.upper() else "PASS" if "PASS" in raw.upper() else None
        gt = "FAIL"  # All images have violations by design

        if pred is not None:
            total += 1
            if pred == gt:
                correct += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/100] detection rate: {correct}/{total}")

    print(f"\nViolation detection: {correct}/{total} ({correct / max(total, 1) * 100:.1f}%)")
    print(f"(All images have violations, so FAIL is always correct)")

    # Continuous test
    print("\n--- Total Violation Amount ---")
    maes: list[float] = []
    for i, s in enumerate(samples):
        path = str(Path(DATASET_DIR) / "test" / f"image_{s['idx']:04d}.png")
        prompt = prompt_total.format(spec=s["spec"]["spec_text"])
        raw = infer(model, processor, path, prompt)
        m = re.search(r'[\d]+\.?\d*', raw)
        pred = float(m.group(0)) if m else None
        if pred is not None:
            maes.append(abs(pred - s["total_violation_mm"]))

        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/100] MAE: {np.mean(maes):.2f}mm")

    gt_totals = [s["total_violation_mm"] for s in samples]
    mean_guess_mae = np.mean([abs(t - np.mean(gt_totals)) for t in gt_totals])

    print(f"\nTotal violation MAE: {np.mean(maes):.2f}mm")
    print(f"Mean guess MAE: {mean_guess_mae:.2f}mm")

    # Print some examples
    print("\n--- Sample Violations ---")
    for s in samples[:5]:
        v_str = "; ".join(
            f"{v.get('hole', v.get('holes', '?'))}: {v['type']} by {v['violation_mm']}mm"
            for v in s["violations"]
        )
        print(f"  Image {s['idx']}: {s['n_holes']} holes, {len(s['violations'])} violations: {v_str}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    if args.generate:
        generate_dataset()
    if args.baseline:
        run_baseline()
    if not (args.generate or args.baseline):
        parser.print_help()


if __name__ == "__main__":
    main()
