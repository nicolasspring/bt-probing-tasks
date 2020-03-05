#!/bin/bash

SBT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SBT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

module load generic
# extracts back-translations from the fairseq output and combines the shards
sbatch -D $REPO -o slurm-%j-extract-beam-bt.out $SBT/job-extract-beam-bt.sh $REPO
