#!/bin/bash

SCRIPTS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO=`dirname "$SCRIPTS"`

# needed for model training
pip install torch fastBPE sacremoses subword_nmt

# needed for probing tasks
pip install numpy scipy scikit-learn matplotlib pandas

# installing fairseq
mkdir -p $REPO/software
cd $REPO/software
git clone https://github.com/nicolasspring/fairseq-states
cd fairseq-states
pip install --editable .
cd ..

# downloading noisy-text
git clone https://github.com/valentinmace/noisy-text
