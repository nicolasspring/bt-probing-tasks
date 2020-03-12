#!/bin/bash

SPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SPT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

BIBEAM=$REPO/model_states/en_de_parallel_plus_bt_beam
BINOISED=$REPO/model_states/en_de_parallel_plus_bt_noised

OUT_PROBING_BEAM=$REPO/probing_tasks/genuine_vs_beamBT
OUT_PROBING_NOISED=$REPO/probing_tasks/genuine_vs_noisedBT

mkdir -p $REPO/probing_tasks/genuine_vs_beamBT
mkdir -p $REPO/probing_tasks/genuine_vs_noisedBT

module load generic
# runs the two probing tasks
sbatch -D $REPO -o slurm-%j-run-probing-tasks.out $SBT/job-run-probing-tasks.sh $REPO
