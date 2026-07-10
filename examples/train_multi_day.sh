#!/bin/bash
# =============================================================================
# Multi-Day Incremental HashEmb Training
# =============================================================================
# Trains day-by-day: each day resumes from the previous day's checkpoint.
# Corresponds to the recommendation-system pattern: train on day T,
# validate on day T's holdout, then continue to day T+1.
#
# Directory structure:
#   $BASE_DIR/
#     20250101/train/*.parquet    ← training data
#     20250101/val/*.parquet      ← validation data (optional)
#     20250102/train/*.parquet
#     20250102/val/*.parquet
#     ...
#     checkpoints/                ← auto-created, stores ckpt_*.pt
#
# Usage:
#   bash examples/train_multi_day.sh --base-dir /data/oss --days 20250101,20250102,20250103
#
#   # Custom hyperparams
#   bash examples/train_multi_day.sh \
#       --base-dir /data/oss \
#       --days 20250101,20250102,20250103 \
#       --steps 5 \
#       --lr 0.01 \
#       --capacity 20000000
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/benchmark_real_data.py"

# ── Default hyperparams (override with CLI) ──
BATCH_SIZE=4096
MAX_RECORDS=0          # 0 = all records
STEPS=3                # epochs per day
LR=0.01
EMBEDDING_DIM=16
CAPACITY=20_000_000
BLOCK_SIZE=1_000_000
PARQUET_BATCH_SIZE=16384
NUM_WORKERS=4
PREFETCH_FACTOR=8
LOG_INTERVAL=50
STATS_SAMPLES=10000
DEBUG=0

# ── Required: --base-dir and --days ──
BASE_DIR=""
DAYS_STR=""
CKPT_DIR=""

usage() {
    cat <<EOF
Usage: $0 --base-dir <DIR> --days <day1,day2,...> [options]

Options:
  --base-dir DIR         Root directory with day subfolders
  --days D1,D2,...       Comma-separated day list (e.g. 20250101,20250102)
  --ckpt-dir DIR         Checkpoint directory (default: BASE_DIR/checkpoints)

  --steps N              Epochs per day (default: $STEPS)
  --batch-size N         Batch size (default: $BATCH_SIZE)
  --max-records N        Max records per epoch (default: 0 = unlimited)
  --lr FLOAT             Learning rate (default: $LR)
  --capacity N           Hash table capacity (default: $CAPACITY)
  --block-size N         Block size (default: $BLOCK_SIZE)
  --parquet-batch-size N Rows per Parquet chunk (default: $PARQUET_BATCH_SIZE)
  --num-workers N        DataLoader workers (default: $NUM_WORKERS)
  --prefetch-factor N    DataLoader prefetch (default: $PREFETCH_FACTOR)
  --debug                Enable debug output

Example:
  $0 --base-dir /data/oss --days 20250101,20250102,20250103
  $0 --base-dir /data/oss --days 20250101,20250102 --steps 5 --lr 0.005
EOF
    exit 1
}

# ── Parse CLI ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)          BASE_DIR="$2"; shift 2 ;;
        --days)              DAYS_STR="$2"; shift 2 ;;
        --ckpt-dir)          CKPT_DIR="$2"; shift 2 ;;
        --steps)             STEPS="$2"; shift 2 ;;
        --batch-size)        BATCH_SIZE="$2"; shift 2 ;;
        --max-records)       MAX_RECORDS="$2"; shift 2 ;;
        --lr)                LR="$2"; shift 2 ;;
        --capacity)          CAPACITY="$2"; shift 2 ;;
        --block-size)        BLOCK_SIZE="$2"; shift 2 ;;
        --parquet-batch-size) PARQUET_BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)       NUM_WORKERS="$2"; shift 2 ;;
        --prefetch-factor)   PREFETCH_FACTOR="$2"; shift 2 ;;
        --debug)             DEBUG=1; shift ;;
        -h|--help)           usage ;;
        *) echo "Unknown: $1"; usage ;;
    esac
done

if [[ -z "$BASE_DIR" ]] || [[ -z "$DAYS_STR" ]]; then
    echo "ERROR: --base-dir and --days are required"
    usage
fi

CKPT_DIR="${CKPT_DIR:-$BASE_DIR/checkpoints}"
mkdir -p "$CKPT_DIR"

IFS=',' read -ra DAYS <<< "$DAYS_STR"

echo "=========================================================="
echo "Multi-Day HashEmb Training"
echo "=========================================================="
echo "  Base dir:     $BASE_DIR"
echo "  Days:         ${DAYS[*]}"
echo "  Checkpoint:   $CKPT_DIR"
echo "  Epochs/day:   $STEPS"
echo "  Batch size:   $BATCH_SIZE"
echo "  LR:           $LR"
echo "  Capacity:     $CAPACITY"
echo "  Parquet bs:   $PARQUET_BATCH_SIZE"
echo "  Workers:      $NUM_WORKERS x prefetch=$PREFETCH_FACTOR"
echo "=========================================================="
echo ""

prev_ckpt=""

for i in "${!DAYS[@]}"; do
    day="${DAYS[$i]}"
    day_idx=$((i + 1))
    total_days=${#DAYS[@]}

    echo "===== Day $day ($day_idx/$total_days) $(date '+%H:%M:%S') ====="

    train_data="$BASE_DIR/$day/train"
    val_data="$BASE_DIR/$day/val"
    ckpt_path="$CKPT_DIR/ckpt_${day}.pt"

    # Build base args
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
        --stats-samples "$STATS_SAMPLES"
        --save "$ckpt_path"
    )

    # Validation data (if exists)
    if compgen -G "$val_data/*.parquet" > /dev/null 2>&1; then
        PY_ARGS+=(--val-data "$val_data/*.parquet")
        echo "  Val:  $val_data"
    else
        echo "  Val:  (none found in $val_data)"
    fi

    # Resume from previous checkpoint
    if [[ -n "$prev_ckpt" ]] && [[ -f "$prev_ckpt" ]]; then
        PY_ARGS+=(--resume "$prev_ckpt")
        echo "  Resume: $prev_ckpt"
    else
        echo "  Resume: (cold start)"
    fi

    if [[ "$DEBUG" -eq 1 ]]; then
        PY_ARGS+=(--debug)
    fi

    echo "  Train: $train_data"
    echo "  Save:  $ckpt_path"
    echo ""

    python "${PY_ARGS[@]}" || {
        echo "ERROR: Day $day failed with exit code $?"
        exit 1
    }

    prev_ckpt="$ckpt_path"
    echo ""
done

echo "=========================================================="
echo "All $total_days days complete  ($(date '+%H:%M:%S'))"
echo "Checkpoints:"
ls -lh "$CKPT_DIR/ckpt_"*.pt 2>/dev/null || echo "  (none)"
echo "=========================================================="
