#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:Tesla-V100:1
#SBATCH --qos=vesta
#SBATCH --partition=volta

# calling script needs to set:
# $REPO

REPO=$1

cd $REPO

BITEXT=$REPO/probing_task_data/bitext
BEAM=$REPO/probing_task_data/beam
NOISED=$REPO/probing_task_data/noised

CHECKPOINT_DIR_BEAM=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_beam
CHECKPOINT_DIR_NOISED=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_noised

BIBEAM=$REPO/model_states/en_de_parallel_plus_bt_beam
BINOISED=$REPO/model_states/en_de_parallel_plus_bt_noised


# task 1: bitext vs beam back-translation (beamBT)

fairseq-states $BITEXT/train.en --path $CHECKPOINT_DIR_BEAM/checkpoint.avg10.pt \
    --states-dir $BIBEAM/bitext

fairseq-states $BEAM/train.en --path $CHECKPOINT_DIR_BEAM/checkpoint.avg10.pt \
    --states-dir $BIBEAM/beam


# task 2: bitext vs noised back-translation (noisedBT)

fairseq-states $BITEXT/train.en --path $CHECKPOINT_DIR_NOISED/checkpoint.avg10.pt \
    --states-dir $BINOISED/bitext

fairseq-states $NOISED/train.en --path $CHECKPOINT_DIR_NOISED/checkpoint.avg10.pt \
    --states-dir $BINOISED/noised
