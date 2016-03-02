#!/bin/bash
# Usage:
# ./experiments/scripts/default_faster_rcnn_alt_opt.sh GPU NET [--set ...]
# Example:
# ./experiments/scripts/default_faster_rcnn_alt_opt.sh 0 ZF \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400,500,600,700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

LOG="experiments/logs/linemod_largecatb_pose_alt_opt_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_faster_rcnn_alt_opt_linemod_largecatb.py --gpu ${GPU_ID} \
  --net_name ${NET} \
  --weights data/imagenet_models/${NET}.v2.caffemodel \
  --imdb linemod_largecatb_train \
  --cfg experiments/wb_cfgs/linemod_largecatb_pose_alt_opt.yml \
  ${EXTRA_ARGS}

