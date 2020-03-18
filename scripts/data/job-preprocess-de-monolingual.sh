#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=generic

# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/backtranslation/prepare-de-monolingual.sh

# calling script needs to set:
# $REPO

REPO=$1

DATAPREP=$REPO/data_prep
cd $DATAPREP

SCRIPTS=$DATAPREP/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=$DATAPREP/subword-nmt/subword_nmt


BPE_CODE=$DATAPREP/wmt18_en_de/code
SUBSAMPLE_SIZE=25000000
LANG=de


OUTDIR=$DATAPREP/wmt18_${LANG}_mono
orig=orig
tmp=$OUTDIR/tmp
mkdir -p $OUTDIR $tmp

FILES=(
    "news.2007.de.shuffled.gz"
    "news.2008.de.shuffled.gz"
    "news.2009.de.shuffled.gz"
    "news.2010.de.shuffled.gz"
    "news.2011.de.shuffled.gz"
    "news.2012.de.shuffled.gz"
    "news.2013.de.shuffled.gz"
    "news.2014.de.shuffled.v2.gz"
    "news.2015.de.shuffled.gz"
    "news.2016.de.shuffled.gz"
    "news.2017.de.shuffled.deduped.gz"
)


if [ -f $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG} ]; then
    echo "found monolingual sample, skipping shuffle/sample/tokenize"
else
    gzip -c -d -k $(for FILE in "${FILES[@]}"; do echo $orig/$FILE; done) \
    | shuf -n $SUBSAMPLE_SIZE \
    | perl $NORM_PUNC $LANG \
    | perl $REM_NON_PRINT_CHAR \
    | perl $TOKENIZER -threads 8 -a -l $LANG \
    > $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG}
fi


if [ -f $tmp/bpe.monolingual.${SUBSAMPLE_SIZE}.${LANG} ]; then
    echo "found BPE monolingual sample, skipping BPE step"
else
    python $BPEROOT/apply_bpe.py -c $BPE_CODE \
        < $tmp/monolingual.${SUBSAMPLE_SIZE}.${LANG} \
        > $tmp/bpe.monolingual.${SUBSAMPLE_SIZE}.${LANG}
fi


if [ -f $tmp/bpe.monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG} ]; then
    echo "found deduplicated monolingual sample, skipping deduplication step"
else
    python $REPO/software/fairseq-states/examples/backtranslation/deduplicate_lines.py $tmp/bpe.monolingual.${SUBSAMPLE_SIZE}.${LANG} \
    > $tmp/bpe.monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG}
fi


if [ -f $OUTDIR/bpe.monolingual.dedup.000.de ]; then
    echo "found sharded data, skipping sharding step"
else
    split --lines 100000 --numeric-suffixes -a 3 \
        --additional-suffix .${LANG} \
        $tmp/bpe.monolingual.dedup.${SUBSAMPLE_SIZE}.${LANG} \
        $OUTDIR/bpe.monolingual.dedup.
fi

# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

cd $REPO

TEXT=$OUTDIR

for SHARD in $(seq -f "%03g" 0 242); do \
    fairseq-preprocess \
        --only-source \
        --source-lang de --target-lang en \
        --joined-dictionary \
        --srcdict $REPO/data-bin/wmt18_en_de/dict.de.txt \
        --testpref $TEXT/bpe.monolingual.dedup.${SHARD} \
        --destdir $REPO/data-bin/wmt18_de_mono/shard${SHARD} \
        --workers 8; \
    cp $REPO/data-bin/wmt18_en_de/dict.en.txt $REPO/data-bin/wmt18_de_mono/shard${SHARD}/; \
done
