#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=generic

# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

# calling script needs to set:
# $REPO

REPO=$1

mkdir -p $REPO/backtranslations/beam

BT_OUT=$REPO/backtranslations/beam

python $REPO/software/fairseq-states/examples/backtranslation/extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output $BT_OUT/bt_beam --srclang en --tgtlang de \
    $BT_OUT/beam.shard*.out
