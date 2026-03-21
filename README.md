# VLM Spatial Measurement Benchmark

A benchmark and training environment for quantitative spatial perception and grounded reasoning in vision-language models.

VLMs can identify objects, read text, and reason about concepts — but they cannot reliably extract **numbers** from spatial features. This benchmark measures that gap with shortcut-proof synthetic tasks, diagnostic tools, and a training curriculum that bridges perception and reasoning.

## Key Findings

### 1. The spatial decoding bottleneck

The vision encoder encodes spatial information well (linear probe R² > 0.9). The language model can't decode it. This gap exists across three architectures and multiple task types.

| Task | Probe MAE | Baseline MAE | Gap | Trained MAE |
|------|-----------|-------------|-----|-------------|
| T1: Hole diameter | 4.31 mm | 7.72 mm | 3.41 mm | 1.39 mm |
| T2: Point distance | 6.41 mm | 10.92 mm | 4.51 mm | 1.71 mm* |
| T4: Alignment offset | 1.14 mm | 7.22 mm | 6.08 mm | 0.74 mm |

*Single-task trained. Multi-task result shown for T1 and T4.

Training closes the gap — and beats the probe, meaning it improves representations, not just decoding.

### 2. Pretraining determines what works out of the box

| Task | Baseline matched pair r | Interpretation |
|------|------------------------|----------------|
| T1: Hole diameter | -0.31 | Ignores scale bar |
| T2: Point distance | 0.48 | Partially uses scale bar |
| T3: Gauge reading | **0.996** | Already knows |
| T4: Alignment offset | 0.087 | Ignores scale bar |

Gauge reading works without training because gauges appear in pretraining data. Scale bar measurement doesn't. The bottleneck is skill-specific, not architectural.

### 3. Spatial measurement decomposes into three components

Transfer test (Task 1 trained → Task 2 data):

| Component | Transfers? | Evidence |
|-----------|-----------|----------|
| Scale bar calibration | ✓ | Matched pair r: 0.48 → 0.84 |
| Output range calibration | ✗ (harmful) | T1→T4 MAE doubles: 7.22 → 14.04 |
| Feature localization | ✗ | T1→T2 MAE: 10.92 → 7.81 (partial) |

### 4. Multi-task training solves calibration lock-in

| Task | Baseline | Single-task | Multi-task |
|------|----------|-------------|------------|
| T1: Diameter (3–30 mm) | 7.72 | 1.62 | **1.39** |
| T2: Distance (5–40 mm) | 10.92 | 1.71 | 3.60 |
| T4: Offset (0.5–8 mm) | 7.22 | n/a | **0.74** |

Task 4 went from worse than guessing (7.22 vs mean-guess 1.85) to 0.74 mm.

### 5. Grounded reasoning requires the two-stage output format

Task 5 (compliance checking) requires measuring a hole AND comparing to a spec. The output format determines whether perception pretraining helps or hurts:

**Single output format (just PASS/FAIL):**

| Condition | Accuracy | Spec ignored |
|-----------|----------|-------------|
| A: from scratch | 66.0% | 21/50 |
| B: perception pretrained | 67.5% | 30/50 |

Pretraining makes the model ignore the spec — it trusts its measurement and doesn't check the text.

**Two-stage format (measurement + verdict):**

| Condition | Accuracy | Spec ignored | Measurement MAE |
|-----------|----------|-------------|-----------------|
| A: from scratch | 79.5% | 13/50 | 1.43 mm |
| B: perception pretrained | **82.5%** | **12/50** | **0.98 mm** |

Forcing explicit measurement before judgment connects perception to reasoning. Pretraining now helps.

**Error decomposition:**

| | A (scratch) | B (pretrained) |
|---|---|---|
| Good measurement → correct verdict | **97.3%** | 91.9% |
| Good measurement → wrong verdict | 4/200 | 14/200 |
| Measurement MAE | 1.43 mm | **0.98 mm** |

Pretraining improves measurement but slightly hurts reasoning. The two-stage format is the key variable — it prevents modality dominance regardless of training order.

### 6. RL cannot discover the two-stage format from scratch

GRPO on Task 5 with binary reward (correct PASS/FAIL) collapses into gibberish by step ~220. The vanishing advantages problem: all outputs get -1.0 reward, advantages are zero, the model degenerates. The two-stage output format must be imposed via SFT before RL can improve reasoning.

## Tasks

### Task 1: Hole Diameter
Measure the diameter of a single hole using a scale bar.

### Task 2: Distance Between Points
Measure the Euclidean distance between two marked points. Harder: two features, diagonal distances.

### Task 3: Gauge Reading
Read a circular dial gauge. Pretraining control — VLMs already know this (r=0.996).

### Task 4: Alignment Offset
Measure the vertical offset between two nearly-aligned holes. Precision task: offsets are 0.5–8 mm vs hole diameters of 5–20 mm.

### Task 5: Grounded Compliance
Image + text spec → PASS/FAIL. Requires measuring from the image AND checking against the spec. Includes no-image ablation and reasoning pair diagnostics.

## Anti-Shortcut Design

Every task is verified shortcut-proof. No single feature predicts the answer.

```
Task 1 (N=1000):  pixel diameter alone: r=+0.44 | correct formula: r=+1.00
Task 2 (N=1000):  pixel distance alone:  r=+0.38 | correct formula: r=+1.00
Task 4 (N=1000):  pixel offset alone:    r=+0.68 | correct formula: r=+1.00
Task 5 (N=1000):  text-only heuristic:   51% accuracy (random)
```

### Verification diagnostics

Each task includes built-in verification:

- **Matched pairs**: identical pixel layout, different scale bars → different answers. Tests scale bar usage.
- **No-image ablation** (Task 5): blank image + real spec → should be ~50%. Tests whether model uses text shortcuts.
- **Reasoning pairs** (Task 5): same image, different specs → one PASS, one FAIL. Tests whether model reads the spec.

## Environment Design

Based on our findings, the environment uses a three-phase training protocol:

### Phase 1: Spatial Perception (SFT, multi-task)
Train on Tasks 1 + 2 + 4 simultaneously with answer-only format. The model learns:
- Scale bar calibration (shared across tasks)
- Feature localization (task-specific)
- Full output range (0.5–40 mm, prevents calibration lock-in)

Advance when: matched pair diagnostics confirm scale bar usage (r > 0.8) across all tasks.

### Phase 2: Grounded Reasoning (SFT, two-stage format)
Train on Task 5 with explicit two-stage output: measurement first, then verdict. Starting from Phase 1 checkpoint. The model learns:
- Connect perception to reasoning via explicit measurement
- Compare measurements against text specifications

Advance when: accuracy > 80%, no-image ablation stays ~50%, reasoning pairs > 35/50.

### Phase 3: Reasoning Improvement (GRPO)
RL with binary reward on Task 5, starting from Phase 2 checkpoint. The model already produces well-formatted two-stage outputs — GRPO improves accuracy without collapsing the format.

**Critical design principles:**
- Multi-task training from the start (prevents calibration lock-in)
- Two-stage output format imposed via SFT before RL (RL cannot discover it)
- Shortcut verification at every phase transition (matched pairs, ablations)
- Perception and reasoning trained jointly but with format separation (prevents modality dominance)

## Usage

### Generate datasets

```bash
python3 generate_task1.py --n-train 1000 --n-test 200 --output-dir dataset_task1
python3 generate_task2.py --n-train 1000 --n-test 200 --output-dir dataset_task2
python3 generate_task3.py --n-train 1000 --n-test 200 --output-dir dataset_task3
python3 generate_task4.py --n-train 1000 --n-test 200 --output-dir dataset_task4
python3 generate_task5.py --n-train 1000 --n-test 200 --output-dir dataset_task5
```

Use `--verify-only` to check shortcut statistics without rendering images.

### Multi-task perception training (Phase 1)

```bash
python3 run_next_session.py --probe    # Vision encoder probe
python3 run_next_session.py --train    # Multi-task SFT (Tasks 1+2+4)
python3 run_next_session.py --eval     # Evaluate all tasks
```

### Grounded reasoning (Phase 2)

```bash
python3 train_task5.py --condition B   # From perception checkpoint
python3 train_task5.py --eval B        # Evaluate with all diagnostics
```

### RL improvement (Phase 3)

```bash
python3 grpo_task5.py --condition B    # GRPO from Phase 2 checkpoint
python3 grpo_task5.py --eval B         # Evaluate with strategy analysis
```

## Infrastructure

- **Model**: Qwen2.5-VL-7B-Instruct + LoRA rank 64
- **GPU**: A100 80GB (bf16, `device_map={"": 0}`)
- **Phase 1**: ~90 min, 9000 steps (3 tasks × 1000 × 3 epochs)
- **Phase 2**: ~35 min, 3000 steps
- **Phase 3**: ~4 hrs (GRPO, 4 generations per sample)
- **Cost**: ~$5–15 total on vast.ai

## Repository Structure

```
generate_task1.py       # Hole diameter (Level 1: single measurement)
generate_task2.py       # Point distance (Level 2: two-feature measurement)
generate_task3.py       # Gauge reading (pretraining control)
generate_task4.py       # Alignment offset (Level 3: precision measurement)
generate_task5.py       # Grounded compliance (Level 4: measurement + reasoning)

eval_task2.py           # Task 2 baseline + probe evaluation
eval_task3.py           # Task 3 gauge reading baseline
eval_task4.py           # Task 4 alignment offset baseline

run_next_session.py     # Multi-task training (Tasks 1+2+4)
train_task5.py          # Task 5 SFT (scratch vs pretrained)
grpo_task5.py           # Task 5 GRPO (RL experiment)
```

## Related Work

- **[vlm-spatial-bottleneck](https://github.com/WFJKK/vlm-spatial-bottleneck)**: Research codebase with full analysis (probing, attention, 8 training methods, 3 model architectures)
- **MeasureBench** (Oct 2025): 1D gauge reading with GRPO. We extend to 2D measurement and grounded reasoning.
- **R1-Zero-VSI** (Apr 2025): GRPO for qualitative spatial reasoning. We test quantitative measurement with continuous rewards.
- **SpatialVLM** (CVPR 2024): "Limited spatial reasoning is a limitation in datasets." Our results confirm this.
- **VL-Rethinker**: Identified vanishing advantages in GRPO for VLMs. We observe the same collapse in our GRPO experiment.
- **ACRE** (ICLR 2026): Found reasoning-answer inconsistency in GRPO. Our two-stage format addresses this by forcing explicit measurement.

## Citation

```
@misc{vlm-spatial-measurement-bench,
  author = {Joshua Kames},
  title = {VLM Spatial Measurement Benchmark},
  year = {2026},
  url = {https://github.com/WFJKK/vlm-spatial-measurement-bench}
}
```
