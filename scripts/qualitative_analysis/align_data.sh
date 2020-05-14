#!/bin/bash

SQA="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SQA"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

TRAIN_DATA=$REPO/data_prep/wmt18_en_de
QUALITATIVE_ANALYSIS=$REPO/qualitative_analysis

ALIGNMENT_TRAIN=$REPO/qualitative_analysis/alignment/training_data
ALIGNMENT_FORWARD=$REPO/qualitative_analysis/alignment/forward_model
ALIGNMENT_REVERSE=$REPO/qualitative_analysis/alignment/reverse_model
ALIGNMENT_INPUT=$REPO/qualitative_analysis/alignment/input
ALIGNMENT_OUTPUT=$REPO/qualitative_analysis/alignment/output

mkdir -p $ALIGNMENT_TRAIN $ALIGNMENT_FORWARD $ALIGNMENT_REVERSE $ALIGNMENT_INPUT/tmp $ALIGNMENT_OUTPUT


# copy training files and remove BPE
cat $TRAIN_DATA/train.en | sed "s/\@\@ //g" > $ALIGNMENT_TRAIN/train.tok.en
cat $TRAIN_DATA/train.de | sed "s/\@\@ //g" > $ALIGNMENT_TRAIN/train.tok.de


# create the fast_align input format
paste $ALIGNMENT_TRAIN/train.tok.en $ALIGNMENT_TRAIN/train.tok.de \
| sed "s/\t/ ||| /g" \
> $ALIGNMENT_TRAIN/wmt_parallel_combined.tok.en-de


# preparing english sources
sacrebleu -t wmt17 -l en-de --echo src \
| sacremoses tokenize -a -l en -q \
> $ALIGNMENT_INPUT/tmp/wmt17.tok.en
sacrebleu -t wmt19 -l en-de --echo src \
| sacremoses tokenize -a -l en -q \
> $ALIGNMENT_INPUT/tmp/wmt19.tok.en


# preparing translations
cat $QUALITATIVE_ANALYSIS/newstest2017/newstest2017_tag.bpe.de \
| sed "s/\@\@ //g" \
> $ALIGNMENT_INPUT/tmp/wmt17.tok.tag.de
cat $QUALITATIVE_ANALYSIS/newstest2017/newstest2017_no_tag.bpe.de \
| sed "s/\@\@ //g" \
> $ALIGNMENT_INPUT/tmp/wmt17.tok.no_tag.de
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.bpe.de \
| sed "s/\@\@ //g" \
> $ALIGNMENT_INPUT/tmp/wmt19.tok.tag.de
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.bpe.de \
| sed "s/\@\@ //g" \
> $ALIGNMENT_INPUT/tmp/wmt19.tok.no_tag.de


# combining files
paste $ALIGNMENT_INPUT/tmp/wmt17.tok.en $ALIGNMENT_INPUT/tmp/wmt17.tok.tag.de \
| sed "s/\t/ ||| /g" \
> $ALIGNMENT_INPUT/wmt17.tok.en-de_tag
paste $ALIGNMENT_INPUT/tmp/wmt17.tok.en $ALIGNMENT_INPUT/tmp/wmt17.tok.no_tag.de \
| sed "s/\t/ ||| /g" \
> $ALIGNMENT_INPUT/wmt17.tok.en-de_no_tag
paste $ALIGNMENT_INPUT/tmp/wmt19.tok.en $ALIGNMENT_INPUT/tmp/wmt19.tok.tag.de \
| sed "s/\t/ ||| /g" \
> $ALIGNMENT_INPUT/wmt19.tok.en-de_tag
paste $ALIGNMENT_INPUT/tmp/wmt19.tok.en $ALIGNMENT_INPUT/tmp/wmt19.tok.no_tag.de \
| sed "s/\t/ ||| /g" \
> $ALIGNMENT_INPUT/wmt19.tok.en-de_no_tag


module load hpc
# training alignment model
TRAIN=$(sbatch -D $REPO -o slurm-%j-train-alignment-model.out $SQA/job-train-alignment-model.sh $REPO \
| tee /dev/tty \
| grep -Po "[0-9]+$")

# applying alignment model
sbatch -D $REPO -o slurm-%j-apply-alignment-model.out --dependency=afterok:$TRAIN \
    $SQA/job-apply-alignment-model.sh $REPO
