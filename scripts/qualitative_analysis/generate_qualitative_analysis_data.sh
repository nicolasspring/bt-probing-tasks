#!/bin/bash

SQA="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SQA"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

MOSES=$REPO/data_prep/mosesdecoder

if [ ! -d "$MOSES" ]; then
    git clone https://github.com/moses-smt/mosesdecoder.git $MOSES
fi

QUALITATIVE_ANALYSIS=$REPO/qualitative_analysis
QA_BIN=$QUALITATIVE_ANALYSIS/bin
QA_TEXT=$QUALITATIVE_ANALYSIS/text

mkdir -p $QUALITATIVE_ANALYSIS/{valid,test}/out $QA_BIN/{no_tag,tag} $QA_TEXT/{no_tag,tag}

module load generic
BINARIZATION=$(sbatch -D $REPO -o slurm-%j-generate-qualitative-analysis-data.out \
                $SQA/job-generate-qualitative-analysis-data.sh $REPO \
| tee /dev/tty \
| grep -Po "[0-9]+$")

module load volta cuda/10.0
# translation has to wait for binarization to finish
sbatch -D $REPO -o slurm-%j-translate-qualitative-analysis-data.out --dependency=afterok:$BINARIZATION \
    $SQA/job-translate-qualitative-analysis-data.sh $REPO
