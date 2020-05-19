#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Nicolas Spring

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import torch

from csv import QUOTE_NONNUMERIC
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models import BaseFairseqModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from typing import Iterable, List, Union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-checkpoint', type=str, required=True,
                        help='path of the fairseq model checkpoint')
    parser.add_argument('--bt-file', type=str, required=True,
                        help='file containing the back-translation data')
    parser.add_argument('--genuine-file', type=str, required=True,
                        help='file containing the genuine data')
    parser.add_argument('--clf-linear-averaging', type=str, required=True,
                        help='file containing a pickled linear classifier for averaged states')
    parser.add_argument('--clf-linear-padding', type=str, required=True,
                        help='file containing a pickled linear classifier for padded states')
    parser.add_argument('--clf-nonlinear-averaging', type=str, required=True,
                        help='file containing a pickled non-linear classifier for averaged states')
    parser.add_argument('--clf-nonlinear-padding', type=str, required=True,
                        help='file containing a pickled non-linear classifier for padded states')
    parser.add_argument('--max-len', type=int,
                        help='maximum sequence length for padding concatenated states')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='location for saving experiment files')
    parser.add_argument('--bt-name', type=str, required=True, choices=['beamBT', 'noisedBT'],
                        help='origin of the model states (beamBT/noisedBT)')
    args = parser.parse_args()
    return args


def load_model_from_checkpoint(path: str, states_dir: str) -> GeneratorHubInterface:
    '''
    loads a fairseq model from a checkpoint and returns it.

    Args:
        path        the location of the model checkpoint
        states_dir  the directory for saving extracted model states

    Returns:
        model       GeneratorHubInterface for translating text
    '''
    os.makedirs(states_dir, exist_ok=True)
    dirname, filename = os.path.split(path)
    model = BaseFairseqModel.from_pretrained(dirname, filename,
                                             tokenizer='moses',
                                             bpe='fastbpe',
                                             encoder_states_dir=states_dir)
    return model


def load_sklearn_clf(path: str) -> Union[LogisticRegression, MLPClassifier]:
    '''
    loads an sklearn classifier and returns it

    Args:
        path    the location of the pickled classifier

    Returns:
        clf     the sklearn classifier
    '''
    with open(path, mode='rb') as infile:
        clf = pickle.load(infile)
    return clf


def pad_state(tensor: torch.Tensor, max_len: int) -> torch.Tensor:
    '''
    concatenates and pads a single model state to a specified length (in time steps)

    Args:
        tensor      the input tensor
        max_len     the sequence length to pad the input tensor to

    Returns:
        out_tensor  the concatenated and padded output tensor
    '''
    assert tensor.shape[0] <= max_len, f"Input tensor is too large for maximum length of {max_len}."
    out_tensor = torch.zeros(max_len, tensor.shape[1], tensor.shape[2])
    out_tensor[:tensor.shape[0], :, :] = tensor
    out_tensor = out_tensor.transpose(0, 1).reshape(out_tensor.shape[1], -1)
    return out_tensor


def average_state(tensor: torch.Tensor) -> torch.Tensor:
    '''
    averages the time steps of a single model state

    Args:
        tensor      the input tensor

    Returns:
        out_tensor  the averaged output tensor
    '''
    out_tensor = tensor.mean(0)
    return out_tensor


def save_confusion_matrix(y_true: List[int],
                          y_pred: List[int],
                          display_labels: np.ndarray,
                          outpath: str):
    '''
    plots a confusion matrix and saves it to the specified path

    Args:
        y_true          list of true labels
        y_pred          list of predicted labels
        display_labels  display labels for the plot
        outpath         path for saving the plot

    '''
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels)
    _ = disp.plot(values_format='.4g')
    plt.savefig(outpath, format='pdf', bbox_inches='tight')

def main(args: argparse.Namespace):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    logger = logging.getLogger('evaluate_classifiers.py')
    logger.info(f'Evaluating classifiers with {args.bt_name}')

    # preparing output directories
    bt_out = os.path.join(args.out_dir, f'states_sents_{args.bt_name}')
    genuine_out = os.path.join(args.out_dir, f'states_sents_genuine')
    assert not (os.path.isdir(bt_out) and os.listdir(bt_out)), \
        'output directory for encoder states must be empty'
    assert not (os.path.isdir(genuine_out) and os.listdir(genuine_out)), \
        'output directory for encoder states must be empty'
    os.makedirs(bt_out, exist_ok=True)
    os.makedirs(genuine_out, exist_ok=True)

    # helper function for obtaining string names for classifiers
    clf_name = lambda x: str(x).split('(')[0]

    # loading trained classifiers
    clf_linear_av = load_sklearn_clf(args.clf_linear_averaging)
    clf_linear_pad = load_sklearn_clf(args.clf_linear_padding)
    clf_nonlinear_av = load_sklearn_clf(args.clf_nonlinear_averaging)
    clf_nonlinear_pad = load_sklearn_clf(args.clf_nonlinear_padding)

    # DataFrames for storing results
    results = pd.DataFrame(columns=['classifier', 'states_operation', 'bt', 'accuracy'])
    all_sents = pd.DataFrame(columns=['sent_number',
                                      'sent',
                                      'true_label',
                                      f'{clf_name(clf_linear_av)}_averaging',
                                      f'{clf_name(clf_linear_pad)}_padding',
                                      f'{clf_name(clf_nonlinear_av)}_averaging',
                                      f'{clf_name(clf_nonlinear_pad)}_padding'])

    # keeping track of predictions
    true_labels = [] # at training, bt had label 1, genuine text label 0
    linear_av_preds = []
    linear_pad_preds = []
    nonlinear_av_preds = []
    nonlinear_pad_preds = []

    # evaluating on back-translations
    logger.info(f'evaluating on {args.bt_name}...')
    model = load_model_from_checkpoint(args.model_checkpoint, bt_out)
    with open(args.bt_file) as bt_file:
        for n, line in enumerate(bt_file, start=1):
            true_labels.append(1)
            model.translate([line], states=True)
            averaged_state = average_state(torch.load(os.path.join(bt_out, f'batch-{n}.pt')))
            padded_state = pad_state(torch.load(os.path.join(bt_out, f'batch-{n}.pt')), args.max_len)

            lin_av_pred = clf_linear_av.predict(averaged_state)[0]
            lin_pad_pred = clf_linear_pad.predict(padded_state)[0]
            nonlin_av_pred = clf_nonlinear_av.predict(averaged_state)[0]
            nonlin_pad_pred = clf_nonlinear_pad.predict(padded_state)[0]
            linear_av_preds.append(lin_av_pred)
            linear_pad_preds.append(lin_pad_pred)
            nonlinear_av_preds.append(nonlin_av_pred)
            nonlinear_pad_preds.append(nonlin_pad_pred)

            all_sents.loc[len(all_sents)] = [n,
                                             line,
                                             1,
                                             lin_av_pred,
                                             lin_pad_pred,
                                             nonlin_av_pred,
                                             nonlin_pad_pred]
            if n % 100 == 0:
                logger.info(f'processed {n} {args.bt_name} sents.')
        logger.info(f'processed all {n} {args.bt_name} sents.')

    # evaluating on genuine source text
    logger.info(f'evaluating on genuine source text...')
    model = load_model_from_checkpoint(args.model_checkpoint, genuine_out)
    with open(args.genuine_file) as genuine_file:
        for n, line in enumerate(genuine_file, start=1):
            true_labels.append(0)
            model.translate([line], states=True)
            averaged_state = average_state(torch.load(os.path.join(genuine_out, f'batch-{n}.pt')))
            padded_state = pad_state(torch.load(os.path.join(genuine_out, f'batch-{n}.pt')), args.max_len)

            lin_av_pred = clf_linear_av.predict(averaged_state)[0]
            lin_pad_pred = clf_linear_pad.predict(padded_state)[0]
            nonlin_av_pred = clf_nonlinear_av.predict(averaged_state)[0]
            nonlin_pad_pred = clf_nonlinear_pad.predict(padded_state)[0]
            linear_av_preds.append(lin_av_pred)
            linear_pad_preds.append(lin_pad_pred)
            nonlinear_av_preds.append(nonlin_av_pred)
            nonlinear_pad_preds.append(nonlin_pad_pred)

            all_sents.loc[len(all_sents)] = [n,
                                             line,
                                             0,
                                             lin_av_pred,
                                             lin_pad_pred,
                                             nonlin_av_pred,
                                             nonlin_pad_pred]
            if n % 100 == 0:
                logger.info(f'processed {n} genuine sents.')
        logger.info(f'processed all {n} genuine sents.')

    all_sents = all_sents.sort_values(['sent_number', 'true_label'], ascending=[True, True])
    label_columns = ['true_label',
                     f'{clf_name(clf_linear_av)}_averaging',
                     f'{clf_name(clf_linear_pad)}_padding',
                     f'{clf_name(clf_nonlinear_av)}_averaging',
                     f'{clf_name(clf_nonlinear_pad)}_padding']
    all_sents[label_columns] = all_sents[label_columns].replace({0:'genuine', 1:f'{args.bt_name}'})

    # evaluating predictions
    logger.info(f'calculating accuracies...')
    lin_av_acc = accuracy_score(true_labels, linear_av_preds)
    lin_pad_acc = accuracy_score(true_labels, linear_pad_preds)
    nonlin_av_acc = accuracy_score(true_labels, nonlinear_av_preds)
    nonlin_pad_acc = accuracy_score(true_labels, nonlinear_pad_preds)
    results.loc[len(results)] = [clf_name(clf_linear_av), 'averaging', args.bt_name, lin_av_acc]
    results.loc[len(results)] = [clf_name(clf_linear_pad), 'padding', args.bt_name, lin_pad_acc]
    results.loc[len(results)] = [clf_name(clf_nonlinear_av), 'averaging', args.bt_name, nonlin_av_acc]
    results.loc[len(results)] = [clf_name(clf_nonlinear_pad), 'padding', args.bt_name, nonlin_pad_acc]
    print(results)

    # saving tables
    results.to_csv(os.path.join(args.out_dir, f'results_genuine_vs_{args.bt_name}.csv'), index=False, quoting=QUOTE_NONNUMERIC)
    all_sents.to_csv(os.path.join(args.out_dir, f'all_sents_genuine_vs_{args.bt_name}.csv'), index=False, quoting=QUOTE_NONNUMERIC)

    # saving confusion matrices
    save_confusion_matrix(true_labels,
                          linear_av_preds,
                          clf_linear_av.classes_,
                          os.path.join(args.out_dir, f'{args.bt_name}_confusion_matrix_{clf_name(clf_linear_av)}_averaging.pdf'))
    save_confusion_matrix(true_labels,
                          linear_pad_preds,
                          clf_linear_pad.classes_,
                          os.path.join(args.out_dir, f'{args.bt_name}_confusion_matrix_{clf_name(clf_linear_pad)}_padding.pdf'))
    save_confusion_matrix(true_labels,
                          nonlinear_av_preds,
                          clf_nonlinear_av.classes_,
                          os.path.join(args.out_dir, f'{args.bt_name}_confusion_matrix_{clf_name(clf_nonlinear_av)}_averaging.pdf'))
    save_confusion_matrix(true_labels,
                          nonlinear_pad_preds,
                          clf_nonlinear_pad.classes_,
                          os.path.join(args.out_dir, f'{args.bt_name}_confusion_matrix_{clf_name(clf_nonlinear_pad)}_padding.pdf'))

if __name__ == '__main__':
    args = parse_args()
    main(args)
