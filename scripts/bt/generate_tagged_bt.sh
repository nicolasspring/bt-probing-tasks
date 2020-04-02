#!/bin/bash

SBT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SBT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

mkdir -p $REPO/backtranslations/tagged

BT_IN=$REPO/backtranslations/beam
BT_OUT=$REPO/backtranslations/tagged

# adds <BT> tags to back-translations to create a taggedBT dataset
sed 's/^/<BT> /' $BT_IN/bt_beam.de > $BT_OUT/bt_tagged.de
sed 's/^/<BT> /' $BT_IN/bt_beam.en > $BT_OUT/bt_tagged.en
