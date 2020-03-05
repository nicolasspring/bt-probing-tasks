#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=generic

# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

# calling script needs to set:
# $REPO

REPO=$1

TEXT=$REPO/backtranslations/beam
fairseq-preprocess \
    --source-lang en --target-lang de \
    --joined-dictionary \
    --srcdict $REPO/data-bin/wmt18_en_de/dict.en.txt \
    --trainpref $TEXT/bt_beam \
    --destdir $REPO/data-bin/wmt18_en_de_bt_beam \
    --workers 8
