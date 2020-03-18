#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-V100:1
#SBATCH --qos=vesta
#SBATCH --partition=volta

# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

# calling script needs to set:
# $REPO
# $SHARD

REPO=$1
SHARD=$2

cd $REPO

BT_OUT=$REPO/backtranslations/beam/out
CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_de_en_parallel

fairseq-generate --fp16 \
    $REPO/data-bin/wmt18_de_mono/shard${SHARD} \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --beam 5 \
> $BT_OUT/beam.shard${SHARD}.out
