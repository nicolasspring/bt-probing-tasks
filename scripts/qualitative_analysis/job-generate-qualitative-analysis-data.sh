#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:Tesla-V100:1
#SBATCH --qos=vesta
#SBATCH --partition=volta

# calling script needs to set:
# $REPO

REPO=$1

BPEROOT=$REPO/data_prep/subword-nmt/subword_nmt
CHECKPOINT_DIR=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_tagged

QUALITATIVE_ANALYSIS=$REPO/qualitative_analysis

# newstest2014 without tag
sacrebleu -t wmt14 -l en-de --echo src \
| tee $QUALITATIVE_ANALYSIS/newstest2014/newstest2014_source.postprocessed.en \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
| tee $QUALITATIVE_ANALYSIS/newstest2014/newstest2014_source.bpe.en \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
    -s en -t de \
    --beam 5 --buffer-size 1024 --max-tokens 8000 \
| grep ^H- \
| cut -f 3- \
| tee $QUALITATIVE_ANALYSIS/newstest2014/newstest2014_no_tag.bpe.de \
| sed "s/\@\@ //g" \
| sacremoses detokenize -l de -q \
> $QUALITATIVE_ANALYSIS/newstest2014/newstest2014_no_tag.postprocessed.de

# newstest2014 with tag
sacrebleu -t wmt14 -l en-de --echo src \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
| sed 's/^/<BT> /' \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
    -s en -t de \
    --beam 5 --buffer-size 1024 --max-tokens 8000 \
| grep ^H- \
| cut -f 3- \
| tee $QUALITATIVE_ANALYSIS/newstest2014/newstest2014_tag.bpe.de \
| sed "s/\@\@ //g" \
| sacremoses detokenize -l de -q \
> $QUALITATIVE_ANALYSIS/newstest2014/newstest2014_tag.postprocessed.de


# newstest2017 without tag
sacrebleu -t wmt17 -l en-de --echo src \
| tee $QUALITATIVE_ANALYSIS/newstest2017/newstest2017_source.postprocessed.en \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
| tee $QUALITATIVE_ANALYSIS/newstest2017/newstest2017_source.bpe.en \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
    -s en -t de \
    --beam 5 --buffer-size 1024 --max-tokens 8000 \
| grep ^H- \
| cut -f 3- \
| tee $QUALITATIVE_ANALYSIS/newstest2017/newstest2017_no_tag.bpe.de \
| sed "s/\@\@ //g" \
| sacremoses detokenize -l de -q \
> $QUALITATIVE_ANALYSIS/newstest2017/newstest2017_no_tag.postprocessed.de

# newstest2017 with tag
sacrebleu -t wmt17 -l en-de --echo src \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
| sed 's/^/<BT> /' \
| fairseq-interactive $REPO/data-bin/wmt18_en_de \
    --path $CHECKPOINT_DIR/checkpoint.avg10.pt \
    -s en -t de \
    --beam 5 --buffer-size 1024 --max-tokens 8000 \
| grep ^H- \
| cut -f 3- \
| tee $QUALITATIVE_ANALYSIS/newstest2017/newstest2017_tag.bpe.de \
| sed "s/\@\@ //g" \
| sacremoses detokenize -l de -q \
> $QUALITATIVE_ANALYSIS/newstest2017/newstest2017_tag.postprocessed.de


# newstest2019 without tag
sacrebleu -t wmt19 -l en-de --echo src \
| tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_source.postprocessed.en \
| sacremoses tokenize -a -l en -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
| tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_source.bpe.en \
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

# newstest2019 with tag
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


# copy reference translations
sacrebleu -t wmt14 -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/newstest2014/newstest2014_ht.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/newstest2014/newstest2014_ht.bpe.de

sacrebleu -t wmt17 -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/newstest2017/newstest2017_ht.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/newstest2017/newstest2017_ht.bpe.de

sacrebleu -t wmt19 -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_wmt19.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_wmt19.bpe.de

sacrebleu -t wmt19/google/ar -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_ar.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_ar.bpe.de

sacrebleu -t wmt19/google/arp -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_arp.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_arp.bpe.de

sacrebleu -t wmt19/google/wmtp -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_wmtp.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_wmtp.bpe.de

sacrebleu -t wmt19/google/hqr -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_hqr.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_hqr.bpe.de

sacrebleu -t wmt19/google/hqp -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_hqp.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_hqp.bpe.de

sacrebleu -t wmt19/google/hqall -l en-de --echo ref \
| tee $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_hqall.postprocessed.de \
| sacremoses tokenize -a -l de -q \
| python $BPEROOT/apply_bpe.py -c $REPO/data-bin/wmt18_en_de/code \
> $QUALITATIVE_ANALYSIS/newstest2019/newstest2019_ht_hqall.bpe.de
