#!/bin/bash
#SBATCH --time=10:00:00
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

WMT18_DATA=$REPO/data_prep/wmt18_en_de
CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_de_en_parallel

# binarize train and test set
# (this step is identical for beamBT and noisedBT and thus will only be performed once)

fairseq-preprocess \
    --only-source \
    --source-lang de --target-lang en \
    --joined-dictionary \
    --srcdict $REPO/data-bin/wmt18_en_de/dict.de.txt \
    --testpref $BEAM_OUT/tmp/train \
    --destdir $BEAM_OUT/data-bin/train \
    --workers 1
cp $REPO/data-bin/wmt18_en_de/dict.en.txt $BEAM_OUT/data-bin/train/
cp $BEAM_OUT/data-bin/train/* $NOISED_OUT/data-bin/train/

fairseq-preprocess \
    --only-source \
    --source-lang de --target-lang en \
    --joined-dictionary \
    --srcdict $REPO/data-bin/wmt18_en_de/dict.de.txt \
    --testpref $BEAM_OUT/tmp/test \
    --destdir $BEAM_OUT/data-bin/test \
    --workers 1
cp $REPO/data-bin/wmt18_en_de/dict.en.txt $BEAM_OUT/data-bin/test/
cp $BEAM_OUT/data-bin/test/* $NOISED_OUT/data-bin/test/

# generating back-translations for the train and test set
# (this step is identical for beamBT and noisedBT and thus will only be performed once)

fairseq-generate --fp16 \
    $BEAM_OUT/data-bin/train \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --beam 5 \
> $BEAM_OUT/tmp/bt_train.out
cp $BEAM_OUT/tmp/bt_train.out $NOISED_OUT/tmp/bt_train.out

fairseq-generate --fp16 \
    $BEAM_OUT/data-bin/test \
    --path $CHECKPOINT_DIR/checkpoint_best.pt \
    --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --beam 5 \
> $BEAM_OUT/tmp/bt_test.out
cp $BEAM_OUT/tmp/bt_test.out $NOISED_OUT/tmp/bt_test.out

# extracting backtranslations from fairseq output
# (this step is identical for beamBT and noisedBT and thus will only be performed once)

python $REPO/software/fairseq-states/examples/backtranslation/extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output $BEAM_OUT/tmp/bt/bt_beam_train --srclang en --tgtlang de \
    $BEAM_OUT/tmp/bt_train.out
cp $BEAM_OUT/tmp/bt/bt_beam_train.en $BEAM_OUT/train.en
cp $BEAM_OUT/tmp/bt/bt_beam_train.en $NOISED_OUT/tmp/bt/bt_beam_train.en

python $REPO/software/fairseq-states/examples/backtranslation/extract_bt_data.py \
    --minlen 1 --maxlen 250 --ratio 1.5 \
    --output $BEAM_OUT/tmp/bt/bt_beam_test --srclang en --tgtlang de \
    $BEAM_OUT/tmp/bt_test.out
cp $BEAM_OUT/tmp/bt/bt_beam_test.en $BEAM_OUT/test.en
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
