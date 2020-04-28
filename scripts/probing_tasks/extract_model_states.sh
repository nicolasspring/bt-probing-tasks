#!/bin/bash

SPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SPT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

BIBEAM=$REPO/model_states/en_de_parallel_plus_bt_beam
BINOISED=$REPO/model_states/en_de_parallel_plus_bt_noised

mkdir -p $REPO/model_states/en_de_parallel_plus_bt_beam/{bitext,beam}/{train,test}
mkdir -p $REPO/model_states/en_de_parallel_plus_bt_noised/{bitext,noised}/{train,test}

module load volta cuda/10.0
# extracts model states for probing task data
sbatch -D $REPO -o slurm-%j-extract-model-states.out $SPT/job-extract-model-states.sh $REPO
