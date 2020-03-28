#!/bin/bash

SEVAL="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SEVAL"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

module load volta cuda/10.0
# evaluate the beam model (en-de)
sbatch -D $REPO -o slurm-%j-evaluate-beam-model.out $SEVAL/job-evaluate-beam-model.sh $REPO
