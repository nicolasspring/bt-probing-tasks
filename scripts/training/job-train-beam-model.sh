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

CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_beam

fairseq-train --fp16 \
    $REPO/data-bin/wmt18_en_de_para_plus_beam \
    --upsample-primary 16 \
    --source-lang en --target-lang de \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 3584 --update-freq 16 \
    --max-update 100000 \
    --save-dir $CHECKPOINT_DIR

python $REPO/software/fairseq-states/scripts/average_checkpoints.py \
    --inputs $CHECKPOINT_DIR \
    --num-epoch-checkpoints 10 \
    --output $CHECKPOINT_DIR/checkpoint.avg10.pt