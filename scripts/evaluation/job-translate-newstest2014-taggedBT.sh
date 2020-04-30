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

BPEROOT=$REPO/data_prep/subword-nmt/subword_nmt
CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_tagged

echo "Calculating (detokenized) sacrebleu on newstest2014 without tag..."
bash $REPO/software/fairseq-states/examples/backtranslation/sacrebleu.sh \
    wmt14 \
    en-de \
    $REPO/data-bin/wmt18_en_de \
    $REPO/data-bin/wmt18_en_de/code \
    $CHECKPOINT_DIR/checkpoint.avg10.pt

echo "Calculating (detokenized) sacrebleu on newstest2014 with an added <BT> tag..."
sacrebleu -t wmt14 -l en-de --echo src \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
| sed 's/^/<BT> /' \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
    -s en -t de \
    --beam 5 --remove-bpe --buffer-size 1024 --max-tokens 8000 \
| grep ^H- \
| cut -f 3- \
| sacremoses detokenize -l de -q \
| sacrebleu -t wmt14 -l en-de
