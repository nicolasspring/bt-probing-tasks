#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-V100:1
#SBATCH --qos=vesta
#SBATCH --partition=volta

# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

# calling script needs to set:
# $REPO

REPO=$1

cd $REPO

BEAM_OUT=$REPO/probing_task_data/beam
NOISED_OUT=$REPO/probing_task_data/noised

CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_de_en_parallel


# back-translating training set

cat $BEAM_OUT/tmp/train.de \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    -s de -t en \
    --beam 5 --buffer-size 1024 --max-tokens 8000 \
| grep -P '^H-' \
| awk -F'H-' '{print $2}' \
| sort -n \
| cut -f 3 \
> $BEAM_OUT/tmp/bt/bt_beam_train.en
# this is the final step for the beam search dataset
cp $BEAM_OUT/tmp/bt/bt_beam_train.en $BEAM_OUT/train.en
# for the noised dataset, noise will be applied after
cp $BEAM_OUT/tmp/bt/bt_beam_train.en $NOISED_OUT/tmp/bt/bt_beam_train.en


# back-translating test set

cat $BEAM_OUT/tmp/test.de \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    -s de -t en \
    --beam 5 --buffer-size 1024 --max-tokens 8000 \
| grep -P '^H-' \
| awk -F'H-' '{print $2}' \
| sort -n \
| cut -f 3 \
> $BEAM_OUT/tmp/bt/bt_beam_test.en
# this is the final step for the beam search dataset
cp $BEAM_OUT/tmp/bt/bt_beam_test.en $BEAM_OUT/test.en
# for the noised dataset, noise will be applied after
cp $BEAM_OUT/tmp/bt/bt_beam_test.en $NOISED_OUT/tmp/bt/bt_beam_test.en


# adding noise to create noisedBT train and test sets

python $REPO/software/noisy-text/add_noise.py $NOISED_OUT/tmp/bt/bt_beam_train.en \
    --output $NOISED_OUT/train.en \
    --delete_probability 0.1 \
    --replace_probability 0.1 \
    --permutation_range 3 \
    --filler_token BLANK

python $REPO/software/noisy-text/add_noise.py $NOISED_OUT/tmp/bt/bt_beam_test.en \
    --output $NOISED_OUT/test.en \
    --delete_probability 0.1 \
    --replace_probability 0.1 \
    --permutation_range 3 \
    --filler_token BLANK
