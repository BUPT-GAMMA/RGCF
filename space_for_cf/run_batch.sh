#!/usr/bin/env bash

DIR=design
CONFIG=cf
GRID=cf
SAMPLE_ALIAS=cf
REPEAT=1
SAMPLE_NUM=3400
MAX_JOBS=1
SLEEP=5
GPU_STRATEGY=greedy

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget


#python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
#  --grid grids/${DIR}/${GRID}.txt \
#  --sample_num $SAMPLE_NUM \
#  --out_dir configs

python configs_gen.py --config configs/${DIR}/${CONFIG}.yaml \
  --grid grids/${DIR}/${GRID}.txt \
  --sample_alias sample/${SAMPLE_ALIAS}.txt \
  --sample_num $SAMPLE_NUM \
  --out_dir configs


python agg_batch.py --dir results/${CONFIG}_grid_${GRID} --metric recall
