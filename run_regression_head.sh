#!/bin/bash
# run_regression_head.sh
#
# Run on vast.ai with A100 80GB.
# NOT Blackwell/RTX PRO 6000 (FP8+LoRA breaks on those).
#
# Usage:
#   bash run_regression_head.sh          # Full run (small test + all 3 modes)
#   bash run_regression_head.sh test     # Small test only (5 samples)
#   bash run_regression_head.sh sft3     # Just sft3_head_only (most interesting)
#   bash run_regression_head.sh resume   # Resume interrupted training
#
# Cost estimate (A100 80GB ~$1.50/hr on vast.ai):
#   - Setup + dataset gen: ~5 min
#   - base_head_only (3 epochs): ~20 min (forward pass cached, only head trains)
#   - base_lora_head (3 epochs): ~90 min (full backward through LoRA)
#   - sft3_head_only (3 epochs): ~20 min (forward pass cached, only head trains)
#   - Evaluation (all 3): ~15 min
#   Total: ~2.5 hours => ~$4
#
# Disk space: ~25GB (model weights ~15GB + dataset ~1GB + checkpoints ~5GB)

set -e

MODE="${1:-full}"

echo "============================================"
echo "  Regression Head Experiment"
echo "  Mode: $MODE"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Disk free: $(df -h /workspace 2>/dev/null | tail -1 | awk '{print $4}' || echo 'unknown')"
echo "============================================"

# --- Safety checks ---
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "")
if echo "$GPU_NAME" | grep -qi "blackwell\|RTX PRO 6000"; then
    echo "ERROR: Blackwell/RTX PRO 6000 detected. FP8+LoRA breaks on these GPUs."
    echo "Please use an A100 80GB instance."
    exit 1
fi

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "0")
if [ "$VRAM_MB" -lt 70000 ] 2>/dev/null; then
    echo "WARNING: Less than 70GB VRAM detected ($VRAM_MB MB)."
    echo "A100 80GB recommended. base_lora_head may OOM."
fi

# --- Setup ---
cd /workspace

if [ ! -d "vlm-spatial-bottleneck" ]; then
    echo "Cloning repo..."
    git clone https://github.com/WFJKK/vlm-spatial-bottleneck.git
fi

cd vlm-spatial-bottleneck

# Copy new script into repo
if [ -f "/workspace/train_regression_head.py" ]; then
    cp /workspace/train_regression_head.py .
fi

# Install deps if needed
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install --break-system-packages -q \
        transformers accelerate peft bitsandbytes \
        pillow numpy scipy scikit-learn qwen-vl-utils
fi

# Generate dataset if needed
if [ ! -d "dataset/train" ]; then
    echo "Generating dataset..."
    python3 generate_dataset.py
fi

echo ""
echo "Dataset ready. Train: $(wc -l < dataset/train/metadata.jsonl) samples"
echo "Test:  $(wc -l < dataset/test/metadata.jsonl) samples"
echo ""

# --- Small test: verify forward pass + hidden states work ---
if [ "$MODE" = "test" ] || [ "$MODE" = "full" ]; then
    echo "=== SMALL TEST: Verifying hidden state extraction ==="
    python3 -c "
import torch, json
from pathlib import Path
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
SYSTEM_PROMPT = 'You are measuring a hole in a technical drawing. Use the scale bar to determine the diameter. Respond with ONLY the diameter in mm as a number, nothing else.'
USER_PROMPT = 'What is the diameter of hole H1 in mm?'

print('Loading model...')
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# Test on 3 samples
with open('dataset/test/metadata.jsonl') as f:
    samples = [json.loads(l) for l in f][:3]

for sample in samples:
    image_path = f'dataset/test/image_{sample[\"idx\"]:04d}.png'
    image = Image.open(image_path).convert('RGB')

    messages = [{'role': 'user', 'content': [
        {'type': 'image'},
        {'type': 'text', 'text': f'{SYSTEM_PROMPT}\n\n{USER_PROMPT}'},
    ]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_layer = outputs.hidden_states[-1]
    last_token = last_layer[0, -1, :]

    print(f'  Sample {sample[\"idx\"]}: hidden shape={last_token.shape}, '
          f'norm={last_token.float().norm():.1f}, gt={sample[\"diameter_mm\"]:.1f}mm')

print()
print('Small test PASSED. Hidden states extracted successfully.')
print(f'Hidden dim: {last_token.shape[0]}')
    "

    if [ "$MODE" = "test" ]; then
        echo "Test-only mode, exiting."
        exit 0
    fi
fi

# --- Training ---
if [ "$MODE" = "resume" ]; then
    RESUME_FLAG="--resume"
else
    RESUME_FLAG=""
fi

if [ "$MODE" = "sft3" ]; then
    echo ""
    echo "=== Running sft3_head_only only ==="
    # Need SFT-3epoch checkpoint
    if [ ! -d "checkpoints_sft_3epoch/final" ]; then
        echo "SFT-3epoch checkpoint not found. Training SFT-3epoch first..."
        python3 train_sft_3epoch.py
    fi
    python3 train_regression_head.py --mode sft3_head_only $RESUME_FLAG
    exit 0
fi

# Full run: all three modes in order of informativeness
echo ""
echo "=== Mode 1/3: base_head_only (cheapest, fastest) ==="
echo "  Can the base model's hidden states predict diameter via regression?"
python3 train_regression_head.py --mode base_head_only $RESUME_FLAG

echo ""
echo "=== Mode 2/3: sft3_head_only (most interesting) ==="
echo "  Does SFT-3epoch model 'know' more than it can say in tokens?"
if [ ! -d "checkpoints_sft_3epoch/final" ]; then
    echo "  Training SFT-3epoch first..."
    python3 train_sft_3epoch.py
fi
python3 train_regression_head.py --mode sft3_head_only $RESUME_FLAG

echo ""
echo "=== Mode 3/3: base_lora_head (most expensive) ==="
echo "  Joint LoRA + regression head training"
python3 train_regression_head.py --mode base_lora_head $RESUME_FLAG

echo ""
echo "============================================"
echo "  All experiments complete!"
echo "  Results in: results/regression_*/"
echo "============================================"
