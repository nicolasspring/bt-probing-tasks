#!/bin/bash

SQA="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
SCRIPTS=`dirname "$SQA"`
REPO=`dirname "$SCRIPTS"`

cd $REPO

QUAL_ANALYSIS_DATA=$REPO/qualitative_analysis
DIVERSITY=$REPO/software/diversity

for YEAR in 2014 2017 2019; do
    printf "\n\nnewstest${YEAR}\n----------------------\n"

    # cTTR calculation
    printf "cTTR\n----\n"
    TAG=$(cat $QUAL_ANALYSIS_DATA/newstest${YEAR}/newstest${YEAR}_tag.postprocessed.de | \
          perl $DIVERSITY/scripts/ttr.pl de no_counts \
          $QUAL_ANALYSIS_DATA/newstest${YEAR}/newstest${YEAR}_source.postprocessed.en)
    printf "Tag (as-if-BT):\t$TAG\n"
    NOTAG=$(cat $QUAL_ANALYSIS_DATA/newstest${YEAR}/newstest${YEAR}_no_tag.postprocessed.de | \
            perl $DIVERSITY/scripts/ttr.pl de no_counts \
            $QUAL_ANALYSIS_DATA/newstest${YEAR}/newstest${YEAR}_source.postprocessed.en)
    printf "No tag:\t\t$NOTAG\n"

    # cMTLD calculation
    printf "cMTLD\n-----\n"
    TAG=$(cat $QUAL_ANALYSIS_DATA/newstest${YEAR}/newstest${YEAR}_tag.postprocessed.de | \
          perl $DIVERSITY/scripts/mtld.pl de no_counts \
          $QUAL_ANALYSIS_DATA/newstest${YEAR}/newstest${YEAR}_source.postprocessed.en)
    printf "Tag (as-if-BT):\t$TAG\n"
    NOTAG=$(cat $QUAL_ANALYSIS_DATA/newstest${YEAR}/newstest${YEAR}_no_tag.postprocessed.de | \
            perl $DIVERSITY/scripts/mtld.pl de no_counts \
            $QUAL_ANALYSIS_DATA/newstest${YEAR}/newstest${YEAR}_source.postprocessed.en)
    printf "No tag:\t\t$NOTAG\n"
done
