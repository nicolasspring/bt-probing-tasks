#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=256G
#SBATCH --partition=hpc

# calling script needs to set:
# $REPO

REPO=$1

BIBEAM=$REPO/model_states/en_de_parallel_plus_bt_beam
BINOISED=$REPO/model_states/en_de_parallel_plus_bt_noised

BITEXT=$REPO/probing_task_data/bitext
BEAM=$REPO/probing_task_data/beam
NOISED=$REPO/probing_task_data/noised

CHECKPOINT_DIR_BEAM=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_beam
CHECKPOINT_DIR_NOISED=$REPO/checkpoints/checkpoints_en_de_parallel_plus_bt_noised

OUT_PROBING_BEAM=$REPO/probing_tasks/genuine_vs_beamBT
OUT_PROBING_NOISED=$REPO/probing_tasks/genuine_vs_noisedBT

echo "Training genuine vs beamBT"
python -u $REPO/scripts/probing_tasks/train_models.py \
        --genuine $BIBEAM/bitext/ \
        --bt $BIBEAM/beam/ \
        --bt-name beamBT \
        --out-dir $OUT_PROBING_BEAM

echo "Training genuine vs noisedBT"
python -u $REPO/scripts/probing_tasks/train_models.py \
        --genuine $BINOISED/bitext/ \
        --bt $BINOISED/noised/ \
        --bt-name noisedBT \
        --out-dir $OUT_PROBING_NOISED

echo "Evaluating genuine vs beamBT"
python -u $REPO/scripts/probing_tasks/evaluate_models.py \
    --model-checkpoint $CHECKPOINT_DIR_BEAM/checkpoint.avg10.pt \
    --bt-file $BEAM/test.en \
    --genuine-file $BITEXT/test.en \
    --clf-linear-averaging $OUT_PROBING_BEAM/clf_averaging_beamBT_LogisticRegression_fitted.pkl \
    --clf-nonlinear-averaging $OUT_PROBING_BEAM/clf_averaging_beamBT_MLPClassifier_fitted.pkl \
    --clf-linear-padding $OUT_PROBING_BEAM/clf_padding_beamBT_LogisticRegression_fitted.pkl \
    --clf-nonlinear-padding $OUT_PROBING_BEAM/clf_padding_beamBT_MLPClassifier_fitted.pkl \
    --max-len 246 \
    --out-dir $OUT_PROBING_BEAM \
    --bt-name beamBT

echo "Evaluating genuine vs noisedBT"
python -u $REPO/scripts/probing_tasks/evaluate_models.py \
    --model-checkpoint $CHECKPOINT_DIR_NOISED/checkpoint.avg10.pt \
    --bt-file $NOISED/test.en \
    --genuine-file $BITEXT/test.en \
    --clf-linear-averaging $OUT_PROBING_NOISED/clf_averaging_noisedBT_LogisticRegression_fitted.pkl \
    --clf-nonlinear-averaging $OUT_PROBING_NOISED/clf_averaging_noisedBT_MLPClassifier_fitted.pkl \
    --clf-linear-padding $OUT_PROBING_NOISED/clf_padding_noisedBT_LogisticRegression_fitted.pkl \
    --clf-nonlinear-padding $OUT_PROBING_NOISED/clf_padding_noisedBT_MLPClassifier_fitted.pkl \
    --max-len 246 \
    --out-dir $OUT_PROBING_NOISED \
    --bt-name noisedBT
