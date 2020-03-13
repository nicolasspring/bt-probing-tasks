#!/bin/bash

SDATA="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SDATA"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

# preprocessing and binarizing the downloaded data

module load generic
WMTPREP=$(sbatch -D $REPO -o slurm-%j-preprocess-wmt18en2de.out $SDATA/job-preprocess-wmt18en2de.sh $REPO \
| tee /dev/tty \
| grep -Po "[0-9]+$")
# preprocessing of monolingual data is dependent on first job
sbatch -D $REPO -o slurm-%j-preprocess-de-monolingual.out --dependency=afterok:$WMTPREP \
    $SDATA/job-preprocess-de-monolingual.sh $REPO
