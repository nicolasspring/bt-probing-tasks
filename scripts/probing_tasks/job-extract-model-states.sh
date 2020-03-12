#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-K80:2
#SBATCH --qos=vesta
#SBATCH --partition=vesta

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
    --states-dir $BIBEAM/bitext/train

fairseq-states $BITEXT/test.en --path $CHECKPOINT_DIR_BEAM/checkpoint.avg10.pt \
    --states-dir $BIBEAM/bitext/test

fairseq-states $BEAM/train.en --path $CHECKPOINT_DIR_BEAM/checkpoint.avg10.pt \
    --states-dir $BIBEAM/beam/train

fairseq-states $BEAM/test.en --path $CHECKPOINT_DIR_BEAM/checkpoint.avg10.pt \
    --states-dir $BIBEAM/beam/test

# task 2: bitext vs noised back-translation (noisedBT)

fairseq-states $BITEXT/train.en --path $CHECKPOINT_DIR_NOISED/checkpoint.avg10.pt \
    --states-dir $BINOISED/bitext/train

fairseq-states $BITEXT/test.en --path $CHECKPOINT_DIR_NOISED/checkpoint.avg10.pt \
    --states-dir $BINOISED/bitext/test

fairseq-states $NOISED/train.en --path $CHECKPOINT_DIR_NOISED/checkpoint.avg10.pt \
    --states-dir $BINOISED/noised/train

fairseq-states $NOISED/test.en --path $CHECKPOINT_DIR_NOISED/checkpoint.avg10.pt \
    --states-dir $BINOISED/noised/test
