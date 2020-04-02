#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=generic

# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

# calling script needs to set:
# $REPO

REPO=$1

BT_OUT=$REPO/backtranslations/beam/out
EXTRACTED=$REPO/backtranslations/beam

python $REPO/software/fairseq-states/examples/backtranslation/extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output $EXTRACTED/bt_beam --srclang en --tgtlang de \
    $BT_OUT/beam.shard*.out
