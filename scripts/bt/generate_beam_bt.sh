#!/bin/bash

SBT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SBT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

mkdir -p $REPO/backtranslations/beam

module load volta cuda/10.0
# generates back-translations with beam size 5
sbatch -D $REPO -o slurm-%j-generate-beam-bt.out $SBT/job-generate-beam-bt.sh $REPO
