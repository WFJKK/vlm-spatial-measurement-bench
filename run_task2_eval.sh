#!/bin/bash
set -e

# Task 2 evaluation: distance between two points
# Qwen2.5-VL-7B baseline + probe
# ~2 hours on A100 80GB, no training

echo "=== Task 2: Distance Between Points ==="
echo "Start time: $(date)"

if [ -n "$CONDA_DEFAULT_ENV" ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda deactivate 2>/dev/null || true
fi

CUDA_VISIBLE_DEVICES=0 python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
assert torch.cuda.is_available(), 'No CUDA'
print(f'GPU: {torch.cuda.get_device_name(0)}')
t = torch.zeros(1).cuda(); del t
print('GPU OK')
" || { echo "GPU check failed"; exit 1; }

pip install --break-system-packages -q qwen-vl-utils peft scikit-learn 2>/dev/null || true

if [ ! -d "dataset_task2/train" ]; then
    echo "Generating Task 2 dataset..."
    python3 generate_task2.py --n-train 1000 --n-test 200 --seed 43 --output-dir dataset_task2
fi

echo "=========================================="
echo "  Task 2: Qwen2.5-VL-7B Baseline + Probe"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python3 eval_task2.py --all

git add results/ embeddings/ 2>/dev/null || true
git commit -m "Task 2 baseline: distance between points" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "=== Complete ==="
echo "End time: $(date)"
