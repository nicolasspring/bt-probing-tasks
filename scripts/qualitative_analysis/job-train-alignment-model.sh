#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=hpc

# calling script needs to set:
# $REPO

REPO=$1

FAST_ALIGN=$REPO/software/fast_align/build

ALIGNMENT_TRAIN=$REPO/qualitative_analysis/alignment/training_data
ALIGNMENT_FORWARD=$REPO/qualitative_analysis/alignment/forward_model
ALIGNMENT_REVERSE=$REPO/qualitative_analysis/alignment/reverse_model

echo "training forward model"
OMP_NUM_THREADS=32 $FAST_ALIGN/fast_align \
    -i $ALIGNMENT_TRAIN/wmt_parallel_combined.tok.en-de \
    -d -v -o \
    -p $ALIGNMENT_FORWARD/params.out \
    > $ALIGNMENT_FORWARD/train_alignments.txt \
    2> $ALIGNMENT_FORWARD/params.err
echo "training completed"

echo "training reverse model"
OMP_NUM_THREADS=32 $FAST_ALIGN/fast_align \
    -i $ALIGNMENT_TRAIN/wmt_parallel_combined.tok.en-de \
    -d -v -o -r \
    -p $ALIGNMENT_REVERSE/params.out \
    > $ALIGNMENT_REVERSE/train_alignments.txt \
    2> $ALIGNMENT_REVERSE/params.err
echo "training completed"
