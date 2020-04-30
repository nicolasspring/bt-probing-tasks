#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-V100:1
#SBATCH --qos=vesta
#SBATCH --partition=volta

# calling script needs to set:
# $REPO

REPO=$1

BPEROOT=$REPO/data_prep/subword-nmt/subword_nmt
CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_tagged

QUALITATIVE_ANALYSIS=$REPO/qualitative_analysis

# newstest2014 as validation set without tag
sacrebleu -t wmt14 -l en-de --echo src \
| tee $QUALITATIVE_ANALYSIS/valid/valid_newstest2014_source.postprocessed.en \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
| tee $QUALITATIVE_ANALYSIS/valid/valid_newstest2014_source.bpe.en \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
    -s en -t de \
    --beam 5 --buffer-size 1024 --max-tokens 8000 \
| grep ^H- \
| cut -f 3- \
| tee $QUALITATIVE_ANALYSIS/valid/valid_newstest2014_no_tag.bpe.de \
| sed "s/\@\@ //g" \
| sacremoses detokenize -l de -q \
> $QUALITATIVE_ANALYSIS/valid/valid_newstest2014_no_tag.postprocessed.de

# newstest2014 as validation set with tag
sacrebleu -t wmt14 -l en-de --echo src \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
| sed 's/^/<BT> /' \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
    -s en -t de \
    --beam 5 --buffer-size 1024 --max-tokens 8000 \
| grep ^H- \
| cut -f 3- \
| tee $QUALITATIVE_ANALYSIS/valid/valid_newstest2014_tag.bpe.de \
| sed "s/\@\@ //g" \
| sacremoses detokenize -l de -q \
> $QUALITATIVE_ANALYSIS/valid/valid_newstest2014_tag.postprocessed.de


# newstest2017 test set without tag
sacrebleu -t wmt17 -l en-de --echo src \
| tee $QUALITATIVE_ANALYSIS/test/test_newstest2017_source.postprocessed.en \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
| tee $QUALITATIVE_ANALYSIS/test/test_newstest2017_source.bpe.en \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
    -s en -t de \
    --beam 5 --buffer-size 1024 --max-tokens 8000 \
| grep ^H- \
| cut -f 3- \
| tee $QUALITATIVE_ANALYSIS/test/test_newstest2017_no_tag.bpe.de \
| sed "s/\@\@ //g" \
| sacremoses detokenize -l de -q \
> $QUALITATIVE_ANALYSIS/test/test_newstest2017_no_tag.postprocessed.de

# newstest2017 test set with tag
sacrebleu -t wmt17 -l en-de --echo src \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
| sed 's/^/<BT> /' \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
    -s en -t de \
    --beam 5 --buffer-size 1024 --max-tokens 8000 \
| grep ^H- \
| cut -f 3- \
| tee $QUALITATIVE_ANALYSIS/test/test_newstest2017_tag.bpe.de \
| sed "s/\@\@ //g" \
| sacremoses detokenize -l de -q \
> $QUALITATIVE_ANALYSIS/test/test_newstest2017_tag.postprocessed.de


# copy reference translations
sacrebleu -t wmt14 -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/valid/valid_newstest2014_ht.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/valid/valid_newstest2014_ht.bpe.de

sacrebleu -t wmt17 -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/test/test_newstest2017_ht.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/test/test_newstest2017_ht.bpe.de
