#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-V100:1
#SBATCH --qos=vesta
#SBATCH --partition=volta

# calling script needs to set:
# $REPO

REPO=$1

MOSES=$REPO/data_prep/mosesdecoder
QUALITATIVE_ANALYSIS=$REPO/qualitative_analysis
QA_BIN=$QUALITATIVE_ANALYSIS/bin
QA_TEXT=$QUALITATIVE_ANALYSIS/text

CHECKPOINTS_TAGGED=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_tagged


# generating translations for valid without tag
fairseq-generate $QA_BIN/no_tag \
    --gen-subset valid \
    --path $CHECKPOINTS_TAGGED/checkpoint.avg10.pt \
    --max-tokens 4096 \
    --beam 5 \
| tee $QUALITATIVE_ANALYSIS/valid/out/valid_no_tag.out \
| grep -P '^H-' \
| awk -F'H-' '{print $2}' \
| sort -n \
| cut -f 3 \
| tee $QUALITATIVE_ANALYSIS/valid/valid_no_tag.bpe.de \
| sed "s/\@\@ //g" \
| perl $MOSES/scripts/tokenizer/detokenizer.perl -q \
> $QUALITATIVE_ANALYSIS/valid/valid_no_tag.postprocessed.de


# generating translations for test without tag
fairseq-generate $QA_BIN/no_tag \
    --gen-subset test \
    --path $CHECKPOINTS_TAGGED/checkpoint.avg10.pt \
    --max-tokens 4096 \
    --beam 5 \
| tee $QUALITATIVE_ANALYSIS/test/out/test_no_tag.out \
| grep -P '^H-' \
| awk -F'H-' '{print $2}' \
| sort -n \
| cut -f 3 \
| tee $QUALITATIVE_ANALYSIS/test/test_no_tag.bpe.de \
| sed "s/\@\@ //g" \
| perl $MOSES/scripts/tokenizer/detokenizer.perl -q \
> $QUALITATIVE_ANALYSIS/test/test_no_tag.postprocessed.de


# generating translations for valid with tag
fairseq-generate $QA_BIN/tag \
    --gen-subset valid \
    --path $CHECKPOINTS_TAGGED/checkpoint.avg10.pt \
    --max-tokens 4096 \
    --beam 5 \
| tee $QUALITATIVE_ANALYSIS/valid/out/valid_tag.out \
| grep -P '^H-' \
| awk -F'H-' '{print $2}' \
| sort -n \
| cut -f 3 \
| tee $QUALITATIVE_ANALYSIS/valid/valid_tag.bpe.de \
| sed "s/\@\@ //g" \
| perl $MOSES/scripts/tokenizer/detokenizer.perl -q \
> $QUALITATIVE_ANALYSIS/valid/valid_tag.postprocessed.de


# generating translations for test with tag
fairseq-generate $QA_BIN/tag \
    --gen-subset test \
    --path $CHECKPOINTS_TAGGED/checkpoint.avg10.pt \
    --max-tokens 4096 \
    --beam 5 \
| tee $QUALITATIVE_ANALYSIS/test/out/test_tag.out \
| grep -P '^H-' \
| awk -F'H-' '{print $2}' \
| sort -n \
| cut -f 3 \
| tee $QUALITATIVE_ANALYSIS/test/test_tag.bpe.de \
| sed "s/\@\@ //g" \
| perl $MOSES/scripts/tokenizer/detokenizer.perl -q \
> $QUALITATIVE_ANALYSIS/test/test_tag.postprocessed.de
