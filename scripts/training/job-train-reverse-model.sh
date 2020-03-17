#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-V100-32GB:1
#SBATCH --qos=vesta
#SBATCH --partition=volta

# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

# calling script needs to set:
# $REPO

REPO=$1

cd $REPO

CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_de_en_parallel

fairseq-train --fp16 \
    $REPO/data-bin/wmt18_en_de \
    --source-lang de --target-lang en \
    --arch transformer_wmt_en_de_big --share-all-embeddings \
    --dropout 0.3 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --max-tokens 14336 --update-freq 32 \
    --max-update 30000 \
    --save-dir $CHECKPOINT_DIR

# copy code and files to checkpoint dir for easy loading of model
cp $REPO/data-bin/wmt18_en_de/code $CHECKPOINT_DIR/code
cp $REPO/data-bin/wmt18_en_de/dict.de.txt $CHECKPOINT_DIR/dict.de.txt
cp $REPO/data-bin/wmt18_en_de/dict.en.txt $CHECKPOINT_DIR/dict.en.txt
