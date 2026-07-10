#!/bin/bash

# 环境变量（按需设）
  export TRAIN_BASE_DIR=/path/to/data
  export CKPT_DIR=/path/to/checkpoints


# 调度器逐天调用
bash examples/aha_home_train_single_daily.sh 20260529 v1 10
bash examples/aha_home_train_single_daily.sh 20260530 v1 10
bash examples/aha_home_train_single_daily.sh 20260601 v1 10
bash examples/aha_home_train_single_daily.sh 20260602 v1 10