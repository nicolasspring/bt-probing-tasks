#!/bin/bash

SEVAL="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SEVAL"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

module load volta cuda/10.0
# evaluate the noised model (en-de)
sbatch -D $REPO -o slurm-%j-evaluate-noised-model.out $STRAINING/job-evaluate-noised-model.sh $REPO
