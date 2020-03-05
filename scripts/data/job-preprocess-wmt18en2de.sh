#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=generic

# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/backtranslation/prepare-wmt18en2de.sh

# calling script needs to set:
# $REPO

REPO=$1

CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training-parallel-nc-v13/news-commentary-v13.de-en"
    "rapid2016.de-en"
)

DATAPREP=$REPO/data_prep
cd $DATAPREP

SCRIPTS=$DATAPREP/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=$DATAPREP/subword-nmt/subword_nmt
BPE_TOKENS=32000

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit 1
fi

OUTDIR=$DATAPREP/wmt18_en_de

src=en
tgt=de
lang=en-de
prep=$OUTDIR
tmp=$prep/tmp
orig=orig

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        echo "$orig/$f.$l"
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/test-full/newstest2014-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

TRAIN=$tmp/train.de-en
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $tmp/bpe.$f
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done


# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

cd $REPO

TEXT=$OUTDIR

fairseq-preprocess \
    --joined-dictionary \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir $REPO/data-bin/wmt18_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 8

cp $TEXT/code data-bin/wmt18_en_de/code
