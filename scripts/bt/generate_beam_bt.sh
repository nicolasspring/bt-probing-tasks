#!/bin/bash

SBT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SBT"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

TEXT=$REPO/data_prep/wmt18_de_mono
BT_OUT=$REPO/backtranslations/beam/out

mkdir -p $REPO/backtranslations/beam/out

# Adapted from https://github.com/bricksdont/noise-distill/blob/master/scripts/translation/decode_parallel_generic.sh

module load volta cuda/10.0

for SHARD in $(seq -f "%03g" 0 242); do
    INPUT=$TEXT/bpe.monolingual.dedup.${SHARD}.de
    OUTPUT=$BT_OUT/beam.shard${SHARD}.out

    if [[ -f $OUTPUT ]]; then
        NUM_LINES_INPUT=$(cat $INPUT | wc -l)
        NUM_LINES_OUTPUT=$(awk '/^H-/{hypos++}END{print hypos}' $OUTPUT)

        if [[ $NUM_LINES_INPUT == $NUM_LINES_OUTPUT ]]; then
            echo "chunk $SHARD exists and number of lines is equal to input ($NUM_LINES_INPUT == $NUM_LINES_OUTPUT)."
            continue
        fi
    fi

    # generates back-translations with beam size 5
    sbatch -D $REPO -o slurm-%j-generate-beam-shard-$SHARD.out $SBT/job-generate-beam-bt.sh $REPO $SHARD
done
