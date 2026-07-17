#!/bin/bash

# 环境变量（按需设）
export TRAIN_BASE_DIR=/home/work/data/wtorchs/batchx/sample_data_v001
export CKPT_DIR=/home/work/data/wtorchs/batchx/ckpt


# 调度器逐天调用
bash gce_mmoe_train_single_daily.sh 20260629 v1 prod
bash gce_mmoe_train_single_daily.sh 20260630 v1 prod
bash gce_mmoe_train_single_daily.sh 20260701 v1 prod
bash gce_mmoe_train_single_daily.sh 20260702 v1 prod
bash gce_mmoe_train_single_daily.sh 20260703 v1 prod
bash gce_mmoe_train_single_daily.sh 20260704 v1 prod
bash gce_mmoe_train_single_daily.sh 20260705 v1 prod
bash gce_mmoe_train_single_daily.sh 20260706 v1 prod
bash gce_mmoe_train_single_daily.sh 20260707 v1 prod