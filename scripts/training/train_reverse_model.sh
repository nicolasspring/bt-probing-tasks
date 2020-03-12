#!/bin/bash

STRAINING="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$STRAINING"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

mkdir -p $REPO/checkpoints/checkpoints_de_en_parallel

module load volta cuda/10.0
# trains a reverse model (de-en)
sbatch -D $REPO -o slurm-%j-train-reverse-model.out $STRAINING/job-train-reverse-model.sh $REPO
