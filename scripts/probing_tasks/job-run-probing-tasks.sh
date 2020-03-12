#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --partition=generic

# calling script needs to set:
# $REPO

REPO=$1

BIBEAM=$REPO/model_states/en_de_parallel_plus_bt_beam
BINOISED=$REPO/model_states/en_de_parallel_plus_bt_noised

OUT_PROBING_BEAM=$REPO/probing_tasks/genuine_vs_beamBT
OUT_PROBING_NOISED=$REPO/probing_tasks/genuine_vs_noisedBT

echo "Experiment genuine vs beamBT"
python $REPO/scripts/probing_tasks/run_experiment.py \
        --genuine $BIBEAM/bitext/ \
        --bt $BIBEAM/beam/ \
        --bt-name beamBT \
        --out-dir $OUT_PROBING_BEAM

echo "Experiment genuine vs noisedBT"
python $REPO/scripts/probing_tasks/run_experiment.py \
        --genuine $BINOISED/bitext/ \
        --bt $BINOISED/noised/ \
        --bt-name beamBT \
        --out-dir $OUT_PROBING_NOISED
