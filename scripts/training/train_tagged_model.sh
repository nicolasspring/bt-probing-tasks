#!/bin/bash

STRAINING="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$STRAINING"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

module load vesta cuda/10.0
# trains a taggedBT model (en-de)
sbatch -D $REPO -o slurm-%j-train-tagged-model.out $STRAINING/job-train-tagged-model.sh $REPO
