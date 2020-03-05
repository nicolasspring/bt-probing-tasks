#!/bin/bash

SBT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SBT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

module load generic
# adds noise to back-translations to create a noisedBT dataset
sbatch -D $REPO -o slurm-%j-generate-noised-bt.out $SBT/job-generate-noised-bt.sh $REPO
