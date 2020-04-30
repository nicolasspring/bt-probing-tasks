#!/bin/bash

SPT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SPT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

PT_DATA=$REPO/probing_task_data
BITEXT_OUT=$REPO/probing_task_data/bitext
BEAM_OUT=$REPO/probing_task_data/beam
NOISED_OUT=$REPO/probing_task_data/noised

mkdir -p $PT_DATA/bitext $PT_DATA/{beam,noised}/tmp/bt

WMT18_DATA=$REPO/data_prep/wmt18_en_de
BPEROOT=$REPO/data_prep/subword-nmt/subword_nmt

# size of training set is TRAIN_SIZE*2 (TRAIN_SIZE sentences of bitext and bt each)
TRAIN_SIZE=25000

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

# get bitext testset
sacrebleu -t wmt17 -l en-de --echo src \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $BITEXT_OUT/test.en

# prepare german side of bitext test set for back-translation
sacrebleu -t wmt17 -l en-de --echo ref \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $BEAM_OUT/tmp/test.de
cp $BEAM_OUT/tmp/test.de $NOISED_OUT/tmp/test.de

module load volta cuda/10.0
# creates the back-translated probing task datasets
sbatch -D $REPO -o slurm-%j-create-probing-task-datasets.out $SPT/job-create-probing-task-datasets.sh $REPO
