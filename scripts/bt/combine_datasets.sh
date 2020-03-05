#!/bin/bash

# Adapted from https://github.com/pytorch/fairseq/tree/master/examples/backtranslation

SBT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SBT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

PARA_DATA=$(readlink -f $REPO/data-bin/wmt18_en_de)
BEAM_DATA=$(readlink -f $REPO/data-bin/wmt18_en_de_bt_beam)
NOISED_DATA=$(readlink -f $REPO/data-bin/wmt18_en_de_bt_noised)
TAGGED_DATA=$(readlink -f $REPO/data-bin/wmt18_en_de_bt_tagged)

COMB_DATA_BEAM=$REPO/data-bin/wmt18_en_de_para_plus_beam
COMB_DATA_NOISED=$REPO/data-bin/wmt18_en_de_para_plus_noised
COMB_DATA_TAGGED=$REPO/data-bin/wmt18_en_de_para_plus_tagged

mkdir -p $COMB_DATA_BEAM
mkdir -p $COMB_DATA_NOISED
mkdir -p $COMB_DATA_TAGGED

for LANG in en de; do \
    # symlink parallel + beam data in $REPO/data-bin/wmt18_en_de_para_plus_beam
    ln -s ${PARA_DATA}/dict.$LANG.txt ${COMB_DATA_BEAM}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -s ${PARA_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA_BEAM}/train.en-de.$LANG.$EXT; \
        ln -s ${BEAM_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA_BEAM}/train1.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/valid.en-de.$LANG.$EXT ${COMB_DATA_BEAM}/valid.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/test.en-de.$LANG.$EXT ${COMB_DATA_BEAM}/test.en-de.$LANG.$EXT; \
    done; \
    # symlink parallel + noised data in $REPO/data-bin/wmt18_en_de_para_plus_noised
    ln -s ${PARA_DATA}/dict.$LANG.txt ${COMB_DATA_NOISED}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -s ${PARA_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA_NOISED}/train.en-de.$LANG.$EXT; \
        ln -s ${NOISED_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA_NOISED}/train1.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/valid.en-de.$LANG.$EXT ${COMB_DATA_NOISED}/valid.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/test.en-de.$LANG.$EXT ${COMB_DATA_NOISED}/test.en-de.$LANG.$EXT; \
    done; \
    # symlink parallel + tagged data in $REPO/data-bin/wmt18_en_de_para_plus_tagged
    ln -s ${PARA_DATA}/dict.$LANG.txt ${COMB_DATA_TAGGED}/dict.$LANG.txt; \
    for EXT in bin idx; do \
        ln -s ${PARA_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA_TAGGED}/train.en-de.$LANG.$EXT; \
        ln -s ${TAGGED_DATA}/train.en-de.$LANG.$EXT ${COMB_DATA_TAGGED}/train1.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/valid.en-de.$LANG.$EXT ${COMB_DATA_TAGGED}/valid.en-de.$LANG.$EXT; \
        ln -s ${PARA_DATA}/test.en-de.$LANG.$EXT ${COMB_DATA_TAGGED}/test.en-de.$LANG.$EXT; \
    done; \
done
