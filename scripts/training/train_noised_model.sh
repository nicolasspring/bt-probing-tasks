#!/bin/bash

STRAINING="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$STRAINING"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

mkdir -p $REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_noised

module load vesta cuda/10.0
# trains a noisedBT model (en-de)
sbatch -D $REPO -o slurm-%j-train-noised-model.out $STRAINING/job-train-noised-model.sh $REPO
