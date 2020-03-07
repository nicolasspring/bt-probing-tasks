#!/bin/bash

SCRIPTS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO=`dirname "$SCRIPTS"`

# needed for model training
pip install fastBPE sacremoses subword_nmt

# specific torch version
wget https://download.pytorch.org/whl/cu100/torch-1.3.0%2Bcu100-cp36-cp36m-linux_x86_64.whl
pip install torch-1.3.0+cu100-cp36-cp36m-linux_x86_64.whl
rm torch-1.3.0+cu100-cp36-cp36m-linux_x86_64.whl

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
