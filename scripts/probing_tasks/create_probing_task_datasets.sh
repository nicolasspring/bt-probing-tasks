#!/bin/bash

SPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SPT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

BITEXT_OUT=$REPO/probing_task_data/bitext
BEAM_OUT=$REPO/probing_task_data/beam
NOISED_OUT=$REPO/probing_task_data/noised

mkdir -p $REPO/probing_task_data/bitext
mkdir -p $REPO/probing_task_data/beam/tmp/bt
mkdir -p $REPO/probing_task_data/noised/tmp/bt
mkdir -p $REPO/probing_task_data/beam/data-bin/{train,test}
mkdir -p $REPO/probing_task_data/noised/data-bin/{train,test}

WMT18_DATA=$REPO/data_prep/wmt18_en_de

TRAIN_SIZE=30000

# to make sure different language splits are parallel
# source: https://www.gnu.org/software/coreutils/manual/html_node/Random-sources.html#Random-sources
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

shuf -n $TRAIN_SIZE --random-source=<(get_seeded_random 42) $WMT18_DATA/train.en > $BITEXT_OUT/train.en
shuf -n $TRAIN_SIZE --random-source=<(get_seeded_random 42) $WMT18_DATA/train.de > $BEAM_OUT/tmp/train.de
cp $BEAM_OUT/tmp/train.de $NOISED_OUT/tmp/train.de
cp $WMT18_DATA/test.en $BITEXT_OUT/test.en
cp $WMT18_DATA/test.de $BEAM_OUT/tmp/test.de
cp $WMT18_DATA/test.de $NOISED_OUT/tmp/test.de

module load volta cuda/10.0
# creates the back-translated probing task datasets
sbatch -D $REPO -o slurm-%j-create-probing-task-datasets.out $SBT/job-create-probing-task-datasets.sh $REPO
