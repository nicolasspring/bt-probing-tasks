#!/bin/bash

SCRIPTS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO=`dirname "$SCRIPTS"`

cd $REPO

module load generic anaconda3
conda create --prefix $REPO/venvs/env-bt-probing-tasks python=3.6 pip
