#!/bin/bash
set -e

# Next session: Task 4 probe + multi-task training
# A100 80GB, ~2 hours total
#
# Usage from fresh instance:
#   deactivate
#   git clone https://github.com/WFJKK/vlm-spatial-measurement-bench.git
#   cd vlm-spatial-measurement-bench
#   bash run_session_setup.sh

echo "=== Setup ==="

if [ -n "$CONDA_DEFAULT_ENV" ]; then
    eval "$(conda shell.bash hook)" 2>/dev/null
    conda deactivate 2>/dev/null || true
fi

pip install --break-system-packages -q torch torchvision matplotlib scikit-learn peft qwen-vl-utils huggingface_hub 2>/dev/null || true

git config user.name "WFJKK"
git config user.email "joshua.kames@proton.me"
git remote set-url origin https://github.com/WFJKK/vlm-spatial-measurement-bench.git

echo "=== Generating datasets ==="
python3 generate_task1.py --n-train 1000 --n-test 200 --output-dir dataset_task1
python3 generate_task2.py --n-train 1000 --n-test 200 --output-dir dataset_task2
python3 generate_task4.py --n-train 1000 --n-test 200 --output-dir dataset_task4

echo ""
echo "=========================================="
echo "  Step 1: Task 4 Probe (10 min)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python3 run_next_session.py --probe

echo ""
echo "CHECK THE PROBE RESULT ABOVE."
echo "If probe MAE > 1.7mm: vision encoder cannot see offsets. Consider skipping."
echo "If probe MAE < 1.5mm: bottleneck exists. Training should help."
echo ""
echo "Continuing to training in 10 seconds... (Ctrl+C to abort)"
sleep 10

echo "=========================================="
echo "  Step 2: Multi-task Training (~90 min)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python3 run_next_session.py --train

echo "=========================================="
echo "  Step 3: Evaluate all tasks (30 min)"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python3 run_next_session.py --eval

echo "=== Pushing results ==="
git add run_next_session.py checkpoints_multitask/training_log.jsonl 2>/dev/null || true
git commit -m "Task 4 probe + multi-task training results" 2>/dev/null || true
git push 2>/dev/null || true

echo ""
echo "=== DONE ==="
echo "End time: $(date)"
