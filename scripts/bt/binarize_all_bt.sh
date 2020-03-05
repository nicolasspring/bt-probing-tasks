#!/bin/bash

SBT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SBT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

module load generic
# binarize the beam search dataset
sbatch -D $REPO -o slurm-%j-binarize-beam.out $SBT/job-binarize-beam-bt.sh $REPO
# binarize the noisedBT dataset
sbatch -D $REPO -o slurm-%j-binarize-noised.out $SBT/job-binarize-noised-bt.sh $REPO
# binarize the taggedBT dataset
sbatch -D $REPO -o slurm-%j-binarize-tagged.out $SBT/job-binarize-tagged-bt.sh $REPO
