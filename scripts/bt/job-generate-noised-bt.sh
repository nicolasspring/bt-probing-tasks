#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=generic

# calling script needs to set:
# $REPO

REPO=$1

BT_IN=$REPO/backtranslations/beam
BT_OUT=$REPO/backtranslations/noised

python $REPO/software/noisy-text/add_noise.py $BT_IN/bt_beam.de \
    --output $BT_OUT/bt_noised.de \
    --delete_probability 0.1 \
    --replace_probability 0.1 \
    --permutation_range 3 \
    --filler_token BLANK

python $REPO/software/noisy-text/add_noise.py $BT_IN/bt_beam.en \
    --output $BT_OUT/bt_noised.en \
    --delete_probability 0.1 \
    --replace_probability 0.1 \
    --permutation_range 3 \
    --filler_token BLANK
