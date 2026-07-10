#!/bin/bash
# =============================================================================
# Single-Day Incremental HashEmb Training
# =============================================================================
# Called once per day by external scheduler (Airflow / DolphinScheduler / cron).
# Automatically resumes from the latest checkpoint for the same model.
#
# Usage:
#   bash aha_home_train_single_daily.sh 20260529 v1
#   bash aha_home_train_single_daily.sh 20260530 v1
#   bash aha_home_train_single_daily.sh 20260601 v1
#
# Args:
#   $1  day         — date string, e.g. 20260529
#   $2  model_name  — model identifier, e.g. v1 / v2_test
#
# Env (set before calling):
#   TRAIN_BASE_DIR  — root of data, default: /data/hashemb
#   CKPT_DIR        — checkpoint dir, default: $TRAIN_BASE_DIR/checkpoints
#
# Data layout expected:
#   $TRAIN_BASE_DIR/$day/train/*.parquet
#   $TRAIN_BASE_DIR/$day/val/*.parquet   (optional)
# =============================================================================

set -euo pipefail

DAY="${1:?Usage: $0 <day> <model_name>}"
MODEL_NAME="${2:?Usage: $0 <day> <model_name>}"

TRAIN_BASE_DIR="${TRAIN_BASE_DIR:-/data/hashemb}"
CKPT_DIR="${CKPT_DIR:-$TRAIN_BASE_DIR/checkpoints}"
PYTHON_SCRIPT="$(dirname "$0")/benchmark_real_data.py"

mkdir -p "$CKPT_DIR"

# ── Fixed hyperparams ──
BATCH_SIZE=4096
MAX_RECORDS=0
STEPS=1
LR=0.01
CAPACITY=20_000_000
BLOCK_SIZE=1_000_000
PARQUET_BATCH_SIZE=8192
NUM_WORKERS=10
PREFETCH_FACTOR=16
LOG_INTERVAL=50

train_data="$TRAIN_BASE_DIR/$DAY/train"
val_data="$TRAIN_BASE_DIR/$DAY/val"
ckpt_path="$CKPT_DIR/${MODEL_NAME}_${DAY}.pt"

echo "=========================================================="
echo "HashEmb Daily Training"
echo "=========================================================="
echo "  Day:         $DAY"
echo "  Model:       $MODEL_NAME"
echo "  Train data:  $train_data"
echo "  Save to:     $ckpt_path"
echo "=========================================================="

# ── Build args ──
PY_ARGS=(
    "$PYTHON_SCRIPT"
    --data "$train_data/*.parquet"
    --batch-size "$BATCH_SIZE"
    --max-records "$MAX_RECORDS"
    --steps "$STEPS"
    --lr "$LR"
    --capacity "$CAPACITY"
    --block-size "$BLOCK_SIZE"
    --parquet-batch-size "$PARQUET_BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --prefetch-factor "$PREFETCH_FACTOR"
    --log-interval "$LOG_INTERVAL"
    --save "$ckpt_path"
)

# ── Validation (if exists) ──
if compgen -G "$val_data/*.parquet" > /dev/null 2>&1; then
    PY_ARGS+=(--val-data "$val_data/*.parquet")
    echo "  Val data:    $val_data"
else
    echo "  Val data:    (none)"
fi

# ── Auto-resume: find latest checkpoint for this model before today ──
latest_ckpt=$(ls -t "$CKPT_DIR/${MODEL_NAME}_"*.pt 2>/dev/null | while read -r ckpt; do
    base=$(basename "$ckpt" .pt)
    ckpt_day="${base#${MODEL_NAME}_}"
    if [[ "$ckpt_day" < "$DAY" ]]; then
        echo "$ckpt"
        break
    fi
done)

if [[ -n "$latest_ckpt" ]]; then
    PY_ARGS+=(--resume "$latest_ckpt")
    echo "  Resume from: $latest_ckpt"
else
    echo "  Resume from: (cold start)"
fi

echo ""

# ── Train ──
python "${PY_ARGS[@]}"

echo ""
echo "  Done: $DAY  →  $ckpt_path"
