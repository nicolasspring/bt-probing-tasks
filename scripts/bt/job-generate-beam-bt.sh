#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-K80:2
#SBATCH --qos=vesta
#SBATCH --partition=vesta

# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

# calling script needs to set:
# $REPO

REPO=$1

cd $REPO

BT_OUT=$REPO/backtranslations/beam
CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_de_en_parallel

for SHARD in $(seq -f "%02g" 0 24); do \
    fairseq-generate --fp16 \
        $REPO/data-bin/wmt18_de_mono/shard${SHARD} \
        --path $CHECKPOINT_DIR/checkpoint_best.pt \
        --skip-invalid-size-inputs-valid-test \
        --max-tokens 4096 \
        --beam 5 \
    > $BT_OUT/beam.shard${SHARD}.out; \
done
