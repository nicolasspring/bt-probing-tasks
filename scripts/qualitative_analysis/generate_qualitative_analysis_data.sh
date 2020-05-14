#!/bin/bash

SQA="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SQA"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

QUALITATIVE_ANALYSIS=$REPO/qualitative_analysis

mkdir -p $QUALITATIVE_ANALYSIS/newstest20{14,17,19}

module load volta cuda/10.0
sbatch -D $REPO -o slurm-%j-generate-qualitative-analysis-data.out $SQA/job-generate-qualitative-analysis-data.sh $REPO
