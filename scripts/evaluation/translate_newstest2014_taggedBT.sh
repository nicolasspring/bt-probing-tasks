#!/bin/bash

SEVAL="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SEVAL"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

module load volta cuda/10.0
# evaluate the tagged model (en-de)
sbatch -D $REPO -o slurm-%j-translate-newstest2014-taggedBT.out $SEVAL/job-translate-newstest2014-taggedBT.sh $REPO
