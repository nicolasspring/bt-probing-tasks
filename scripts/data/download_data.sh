#!/bin/bash

SDATA="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SDATA"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

mkdir data_prep
cd data_prep

# downloading wmt18 data
bash $REPO/software/fairseq-states/examples/backtranslation/download-wmt18en2de.sh
# downloading monolingual data for back-translation
bash $REPO/software/fairseq-states/examples/backtranslation/download-de-monolingual.sh
