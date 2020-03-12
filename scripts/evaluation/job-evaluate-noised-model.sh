#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-V100:1
#SBATCH --qos=vesta
#SBATCH --partition=volta

# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

# calling script needs to set:
# $REPO

REPO=$1

cd $REPO

CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_noised

bash $REPO/software/fairseq-states/examples/backtranslation/sacrebleu.sh \
    wmt17 \
    en-de \
    $REPO/data-bin/wmt18_en_de \
    $REPO/data-bin/wmt18_en_de/code \
    $CHECKPOINT_DIR/checkpoint.avg10.pt
