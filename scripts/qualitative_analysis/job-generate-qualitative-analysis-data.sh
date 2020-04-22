#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=generic


# calling script needs to set:
# $REPO

REPO=$1

MOSES=$REPO/data_prep/mosesdecoder

QUALITATIVE_ANALYSIS=$REPO/qualitative_analysis
QA_BIN=$QUALITATIVE_ANALYSIS/bin
QA_TEXT=$QUALITATIVE_ANALYSIS/text
DATA_TXT=$REPO/data_prep/wmt18_en_de
DATA_BIN=$REPO/data-bin/wmt18_en_de


# copy data as is for generation without tag
cp $DATA_TXT/valid.en $QA_TEXT/no_tag/valid_no_tag.en
cp $DATA_TXT/test.en $QA_TEXT/no_tag/test_no_tag.en

# binarizing data without tag
fairseq-preprocess \
    --only-source \
    --source-lang en --target-lang de \
    --joined-dictionary --srcdict $DATA_BIN/dict.en.txt \
    --validpref $QA_TEXT/no_tag/valid_no_tag \
    --testpref $QA_TEXT/no_tag/test_no_tag \
    --destdir $QA_BIN/no_tag \
    --workers 8
cp $DATA_BIN/{code,dict.de.txt} $QA_BIN/no_tag/


# adding <BT> tag for generation with tag
cat $DATA_TXT/valid.en | sed 's/^/<BT> /' > $QA_TEXT/tag/valid_tag.en
cat $DATA_TXT/test.en | sed 's/^/<BT> /' > $QA_TEXT/tag/test_tag.en

# binarizing data with tag
fairseq-preprocess \
    --only-source \
    --source-lang en --target-lang de \
    --joined-dictionary --srcdict $DATA_BIN/dict.en.txt \
    --validpref $QA_TEXT/tag/valid_tag \
    --testpref $QA_TEXT/tag/test_tag \
    --destdir $QA_BIN/tag \
    --workers 8
cp $DATA_BIN/{code,dict.de.txt} $QA_BIN/tag/


# copying the human translations
cp $DATA_TXT/valid.de $QUALITATIVE_ANALYSIS/valid/valid_ht.bpe.de
cp $DATA_TXT/test.de $QUALITATIVE_ANALYSIS/test/test_ht.bpe.de

cat $DATA_TXT/valid.de \
| sed "s/\@\@ //g" \
| perl $MOSES/scripts/tokenizer/detokenizer.perl -q \
> $QUALITATIVE_ANALYSIS/valid/valid_ht.postprocessed.de

cat $DATA_TXT/test.de \
| sed "s/\@\@ //g" \
| perl $MOSES/scripts/tokenizer/detokenizer.perl -q \
> $QUALITATIVE_ANALYSIS/test/test_ht.postprocessed.de

cp $DATA_TXT/valid.en $QUALITATIVE_ANALYSIS/valid/valid_source.bpe.en
cp $DATA_TXT/test.en $QUALITATIVE_ANALYSIS/test/test_source.bpe.en

cat $DATA_TXT/valid.en \
| sed "s/\@\@ //g" \
| perl $MOSES/scripts/tokenizer/detokenizer.perl -q \
> $QUALITATIVE_ANALYSIS/valid/valid_source.postprocessed.en

cat $DATA_TXT/test.en \
| sed "s/\@\@ //g" \
| perl $MOSES/scripts/tokenizer/detokenizer.perl -q \
> $QUALITATIVE_ANALYSIS/test/test_source.postprocessed.en
