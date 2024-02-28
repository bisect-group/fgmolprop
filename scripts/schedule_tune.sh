#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

for dataset in bbbp tox21 toxcast sider clintox
do
    for method in FG MFG FGR
    do
        for descriptors in true false
        do
            CUDA_VISIBLE_DEVICES=0 python src/train.py -m hparams_search=fgr_optuna experiment=$dataset data.method=$method data.descriptors=$descriptors
        done
    done
done
