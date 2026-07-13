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

set -eu

DAY="${1:?Usage: $0 <day> <model_name> <debug_or_prod>}"
MODEL_NAME="${2:?Usage: $0 <day> <model_name> <debug_or_prod>}"
DOP="${3:?Usage: $0 <day> <model_name> <debug_or_prod>}"

TRAIN_BASE_DIR="${TRAIN_BASE_DIR:-/data/hashemb}"
CKPT_DIR="${CKPT_DIR:-$TRAIN_BASE_DIR/checkpoints}"
PYTHON_SCRIPT="$(dirname "$0")/benchmark_real_data.py"

echo "DAY: ${DAY} MODEL_NAME: ${MODEL_NAME} TRAIN_BASE_DIR: ${TRAIN_BASE_DIR} CKPT_DIR: ${CKPT_DIR} PYTHON_SCRIPT: ${PYTHON_SCRIPT}"
dt=${DAY}
model_name=${MODEL_NAME}
base_hdfs_path="oss://transsion-aisearch-hadoop-prod-alifk/aha/tmp/aha_rec_home_sample_tfrd_v013_reflow2_v005_snapy_1bill"
data_hdfs_path="${base_hdfs_path}/${dt}"
echo "data_hdfs_path: $data_hdfs_path"

formatted_dt=$(date -d "${DAY}" "+%Y-%m-%d")
local_data_prefix="${TRAIN_BASE_DIR}/${model_name}"
local_data_path="${local_data_prefix}/${dt}"
local_valid_path="${local_data_prefix}/${formatted_dt}"

# 检查HDFS路径是否存在
if [ $DOP = "debug" ]; then
  echo "debug"
else
    # hadoop fs -ls  "$data_hdfs_path" > /dev/null 2>&1
    if ossutil ls  "$data_hdfs_path" > /dev/null 2>&1; then
        echo "oss path exists, proceeding with data pull..."
        if [ -d ${local_data_path} ]; then
            echo "Removing existing local directory..."
            rm -rf ${local_data_path}
            sleep 25
        fi
        # 创建本地目录
        mkdir -p ${local_data_path}
        # 创建验证目录
        mkdir -p ${local_valid_path}
        # 下载数据
        # hadoop fs -get "$data_hdfs_path/*" ${local_data_path}
        ossutil cp "$data_hdfs_path/" ${local_data_path} -r
        echo "Data downloaded to ${local_data_path}"

        last_file=$(find ${local_data_path} -type f | tail -n 1)
        if [ -n "$last_file" ]; then
            echo "Moving last file to validation directory: $last_file"
            mv "$last_file" ${local_valid_path}/
        else
            echo "No files found to move to validation directory"
        fi


    else
        echo "Error: HDFS path does not exist: $data_hdfs_path"
        exit 1
    fi

    # 检查文件数量是否大于等于100个
    file_count=$(find ${local_data_path} -type f | wc -l)
    echo "Number of files downloaded: $file_count"

    if [ "$file_count" -lt 100 ]; then
        echo "Error: Number of files ($file_count) is less than 100. Exiting..."
        exit 1
    fi

    # 不使用bc的版本（仅适用于整数比较）
    total_size_kb=$(du -sk ${local_data_path} | cut -f1)
    total_size_gb=$((total_size_kb / 1024 / 1024))

    echo "Total size of ${local_data_path}: $((total_size_kb / 1024)) MB (${total_size_gb} GB)"

    if [ $total_size_gb -lt 10 ]; then
        echo "Error: Total size of directory (${total_size_gb} GB) is less than 10 GB. Exiting..."
        exit 1
    fi
fi 
echo "File count check passed. Proceeding with training..."


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

EVICT_MIN_COUNT=20
EVICT_MAX_IDLE_DAYS=3
EVICT_COMBINE=and

# ── Eviction (optional, set via env) ──
EVICT_MIN_COUNT="${EVICT_MIN_COUNT:-0}"
EVICT_MAX_IDLE_DAYS="${EVICT_MAX_IDLE_DAYS:-0}"
EVICT_COMBINE="${EVICT_COMBINE:-and}"

train_data=${local_data_path}
val_data=${local_valid_path}
ckpt_path="$CKPT_DIR/${MODEL_NAME}_${DAY}.pt"

echo "=========================================================="
echo "HashEmb Daily Training"
echo "=========================================================="
echo "  Day:         $DAY"
echo "  Model:       $MODEL_NAME"
echo "  Train data:  $train_data"
echo "  Hash table:  ${ckpt_path%.pt}.hashemb"
echo "  Dense model: $ckpt_path"
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
    --evict-min-count "$EVICT_MIN_COUNT"
    --evict-max-idle-days "$EVICT_MAX_IDLE_DAYS"
    --evict-combine "$EVICT_COMBINE"
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
echo "  Done: $DAY  →  ${ckpt_path%.pt}.hashemb  +  $ckpt_path"

if [ $DOP = "debug" ]; then
    echo "Debug mode enabled."
else
    echo "Training completed, Removing local directory ${local_data_path} ..."
    rm -rf ${local_data_path}
fi