#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=hpc

# calling script needs to set:
# $REPO

REPO=$1

FAST_ALIGN=$REPO/software/fast_align/build

ALIGNMENT_FORWARD=$REPO/qualitative_analysis/alignment/forward_model
ALIGNMENT_REVERSE=$REPO/qualitative_analysis/alignment/reverse_model

ALIGNMENT_INPUT=$REPO/qualitative_analysis/alignment/input
ALIGNMENT_OUTPUT=$REPO/qualitative_analysis/alignment/output

echo "applying to tagged wmt14"
python2 $FAST_ALIGN/force_align.py \
    $ALIGNMENT_FORWARD/params.out \
    $ALIGNMENT_FORWARD/params.err \
    $ALIGNMENT_REVERSE/params.out \
    $ALIGNMENT_REVERSE/params.err \
    grow-diag-final-and \
    < $ALIGNMENT_INPUT/valid_wmt14.tok.en-de_tag \
    > $ALIGNMENT_OUTPUT/valid_wmt14.aligned.en-de_tag

echo "applying to untagged wmt14"
python2 $FAST_ALIGN/force_align.py \
    $ALIGNMENT_FORWARD/params.out \
    $ALIGNMENT_FORWARD/params.err \
    $ALIGNMENT_REVERSE/params.out \
    $ALIGNMENT_REVERSE/params.err \
    grow-diag-final-and \
    < $ALIGNMENT_INPUT/valid_wmt14.tok.en-de_no_tag \
    > $ALIGNMENT_OUTPUT/valid_wmt14.aligned.en-de_no_tag

echo "applying to tagged wmt17"
python2 $FAST_ALIGN/force_align.py \
    $ALIGNMENT_FORWARD/params.out \
    $ALIGNMENT_FORWARD/params.err \
    $ALIGNMENT_REVERSE/params.out \
    $ALIGNMENT_REVERSE/params.err \
    grow-diag-final-and \
    < $ALIGNMENT_INPUT/test_wmt17.tok.en-de_tag \
    > $ALIGNMENT_OUTPUT/test_wmt17.aligned.en-de_tag

echo "applying to untagged wmt17"
python2 $FAST_ALIGN/force_align.py \
    $ALIGNMENT_FORWARD/params.out \
    $ALIGNMENT_FORWARD/params.err \
    $ALIGNMENT_REVERSE/params.out \
    $ALIGNMENT_REVERSE/params.err \
    grow-diag-final-and \
    < $ALIGNMENT_INPUT/test_wmt17.tok.en-de_no_tag \
    > $ALIGNMENT_OUTPUT/test_wmt17.aligned.en-de_no_tag
