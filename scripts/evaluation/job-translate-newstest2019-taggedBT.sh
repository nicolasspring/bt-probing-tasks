#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-V100:1
#SBATCH --qos=vesta
#SBATCH --partition=volta

# calling script needs to set:
# $REPO

REPO=$1

cd $REPO

BPEROOT=$REPO/data_prep/subword-nmt/subword_nmt
CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_tagged
QUALITATIVE_ANALYSIS=$REPO/qualitative_analysis

if [ ! -f $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.postprocessed.de ]; then
    sacrebleu -t wmt19 -l en-de --echo src \
    | sacremoses tokenize -a -l en -q \
    | python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
    | fairseq-interactive $REPO/data-bin/wmt18_en_de \
        --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
        -s en -t de \
        --beam 5 --buffer-size 1024 --max-tokens 8000 \
    | grep ^H- \
    | cut -f 3- \
    | tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.bpe.de \
    | sed "s/\@\@ //g" \
    | sacremoses detokenize -l de -q \
    > $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.postprocessed.de
fi

if [ ! -f $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.postprocessed.de ]; then
    sacrebleu -t wmt19 -l en-de --echo src \
    | sacremoses tokenize -a -l en -q \
    | python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
    | sed 's/^/<BT> /' \
    | fairseq-interactive $REPO/data-bin/wmt18_en_de \
        --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
        -s en -t de \
        --beam 5 --buffer-size 1024 --max-tokens 8000 \
    | grep ^H- \
    | cut -f 3- \
    | tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.bpe.de \
    | sed "s/\@\@ //g" \
    | sacremoses detokenize -l de -q \
    > $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.postprocessed.de
fi

echo "Detokenized BLEU scores:"

echo "no tag:"
echo "wmt19"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.postprocessed.de | sacrebleu -t wmt19 -l en-de
echo "wmt19/google/ar"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.postprocessed.de | sacrebleu -t wmt19/google/ar -l en-de
echo "wmt19/google/arp"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.postprocessed.de | sacrebleu -t wmt19/google/arp -l en-de
echo "wmt19/google/wmtp"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.postprocessed.de | sacrebleu -t wmt19/google/wmtp -l en-de
echo "wmt19/google/hqr"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.postprocessed.de | sacrebleu -t wmt19/google/hqr -l en-de
echo "wmt19/google/hqp"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.postprocessed.de | sacrebleu -t wmt19/google/hqp -l en-de
echo "wmt19/google/hqall"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_no_tag.postprocessed.de | sacrebleu -t wmt19/google/hqall -l en-de

echo "<BT> tag (as-if-BT):"
echo "wmt19"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.postprocessed.de | sacrebleu -t wmt19 -l en-de
echo "wmt19/google/ar"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.postprocessed.de | sacrebleu -t wmt19/google/ar -l en-de
echo "wmt19/google/arp"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.postprocessed.de | sacrebleu -t wmt19/google/arp -l en-de
echo "wmt19/google/wmtp"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.postprocessed.de | sacrebleu -t wmt19/google/wmtp -l en-de
echo "wmt19/google/hqr"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.postprocessed.de | sacrebleu -t wmt19/google/hqr -l en-de
echo "wmt19/google/hqp"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.postprocessed.de | sacrebleu -t wmt19/google/hqp -l en-de
echo "wmt19/google/hqall"
cat $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_tag.postprocessed.de | sacrebleu -t wmt19/google/hqall -l en-de
