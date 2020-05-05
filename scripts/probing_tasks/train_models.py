#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Nicolas Spring

import argparse
import logging
import pandas as pd
import pickle
import os
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from typing import List, Tuple, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--genuine', type=str, required=True,
                        help='directory containing model states of genuine source text')
    parser.add_argument('--bt', type=str, required=True,
                        help='directory containing model states of either beamBT or noisedBT')
    parser.add_argument('--bt-name', type=str, required=True, choices=['beamBT', 'noisedBT'],
                        help='origin of the model states (beamBT/noisedBT)')
    parser.add_argument('--out-dir', type=str, required=True,
                        help='directory for the outputs of the experiment')
    args = parser.parse_args()
    return args


def get_max_seq_len(directories: List[str], logger: logging.Logger) -> int:
    '''
    returns the maximum sequence length in directories containing model states.

    Args:
        directories     list of directory paths
        logger          a logging.Logger instance

    Returns:
        max_len         the maximum sequence length
    '''
    all_files = []
    for directory in directories:
        states_dir, _, filenames = next(os.walk(os.path.abspath(directory)))
        for filename in filenames:
            all_files.append(os.path.join(states_dir, filename))
    all_tensors = (torch.load(filename, map_location=torch.device('cpu')) for filename in all_files)
    max_len = max(all_tensors, key=lambda x: x.shape[0]).shape[0]
    # please compare to the output of scripts/checks/check_state_shapes.py
    logger.info(f'maximum number of time steps is {max_len}')
    return max_len


def pad_states(directory: str, max_len: int, logger: logging.Logger) -> torch.Tensor:
    '''
    reads model states from file and creates a feature matrix with one row per sentence
    by padding all sentences to the maximum time steps.

    Args:
        directory   a directory containing saved model states
        max_len     the maximum sequence time steps
        logger      a logging.Logger instance

    Returns:
        X           the padded feature matrix
    '''
    logger.info(f'reading tensors from {directory}')
    states_dir, _, filenames = next(os.walk(os.path.abspath(directory)))
    tensors = [torch.load(os.path.join(states_dir, filename), map_location=torch.device('cpu'))
               for filename in filenames if filename.startswith('batch') and filename.endswith('.pt')]
    # please compare to the output of scripts/checks/check_state_shapes.py
    logger.info(f'encoder size is {tensors[0].shape[2]}')

    # reshaping and padding individual tensors to max_len
    reshaped_tensors = []
    for tensor in tensors:
        if tensor.shape[0] <= max_len:
            target = torch.zeros(max_len, tensor.shape[1], tensor.shape[2])
            target[:tensor.shape[0], :, :] = tensor # padding with zeros to max(time_steps)
            # creating a tensor with one row per sentence
            reshaped_tensors.append(target.transpose(0, 1).reshape(target.shape[1], -1))

    # concatenating tensors to create a feature matrix
    X = torch.cat(reshaped_tensors, dim=0)
    logger.info(f'created padded feature matrix with shape {X.shape}')
    return X


def average_states(directory: str, logger: logging.Logger) -> torch.Tensor:
    '''
    reads model states from file and creates a feature matrix with one row per sentence
    by averaging the tokens per sentence.

    Args:
        directory   a directory containing saved model states
        logger      a logging.Logger instance

    Returns:
        X           the averaged feature matrix
    '''
    logger.info(f'reading tensors from {directory}')
    states_dir, _, filenames = next(os.walk(os.path.abspath(directory)))
    tensors = [torch.load(os.path.join(states_dir, filename), map_location=torch.device('cpu'))
               for filename in filenames if filename.startswith('batch') and filename.endswith('.pt')]
    # please compare to the output of scripts/checks/check_state_shapes.py
    logger.info(f'encoder size is {tensors[0].shape[2]}')

    # averaging all relevant (no padding) time steps
    tensors = [tensor.mean(0) for tensor in tensors]

    # concatenating tensors to create a feature matrix
    X = torch.cat(tensors, dim=0)
    logger.info(f'created averaged feature matrix with shape {X.shape}')
    return X


def create_dataset(source_matrix: torch.Tensor, bt_matrix: torch.Tensor) -> Tuple[pd.DataFrame, pd.Series]:
    '''
    creates an X matrix and a y vector for sklearn training from two tensors.
    source tensor receives label 0, bt tensor receives label 1.

    Args:
        source_matrix   a torch.Tensor, the feature matrix for genuine text
        bt_matrix       a torch.Tensor, the feature matrix for bt
    '''
    # creating DataFrames from the tensors and adding labels
    source_df = pd.DataFrame(source_matrix.cpu().numpy())
    source_df['label'] = [0] * len(source_df)
    bt_df = pd.DataFrame(bt_matrix.cpu().numpy())
    bt_df['label'] = [1] * len(bt_df)
    
    # combining and shuffling data
    combined = source_df.append(bt_df, ignore_index=True)
    combined = combined.sample(frac=1)

    X = combined.drop(columns=['label'])
    y = combined['label'].copy()

    return X, y


def train_models(X_train: pd.DataFrame,
                 y_train: pd.DataFrame,
                 bt_name: str,
                 experiment: str,
                 out_dir: str,
                 logger: logging.Logger):
    '''
    Fits classifiers for a probing task experiment.

    Args:
        X_train     training features
        y_train     training labels
        bt_name     name of the bt dataset used for naming files
        experiment  the name of the experiment
        out_dir     output directory
        logger      a logging.Logger instance
    '''
    linear_clf_params = {'C': 0.0001,
                         'max_iter': 100,
                         'solver': 'liblinear',
                         'tol': 1e-4,
                         'verbose':100
                         }
    mlp_clf_params = {'activation': 'relu',
                      'alpha': 0.0001,
                      'beta_1': 0.9,
                      'epsilon': 10e-8,
                      'hidden_layer_sizes': (100,),
                      'learning_rate_init': 0.0001,
                      'solver': 'adam',
                      'early_stopping':True,
                      'n_iter_no_change':10,
                      'validation_fraction':0.02,
                      'verbose':True
                      }
    linear_clf = LogisticRegression(**linear_clf_params)
    mlp_clf = MLPClassifier(**mlp_clf_params)

    for clf in [linear_clf, mlp_clf]:
        clf_name = str(clf).split('(')[0]
        logger.info(f'fitting {clf_name} classifier on {bt_name} with {experiment}...')
        clf.fit(X_train, y_train)
        with open(os.path.join(out_dir, f'clf_{experiment}_{bt_name}_{clf_name}_fitted.pkl'), 'wb') as outfile:
            pickle.dump(clf, outfile)


def main(args: argparse.Namespace):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )
    logger = logging.getLogger('train_models.py')

    out_dir = os.path.abspath(args.out_dir)

    max_len = get_max_seq_len([args.genuine, args.bt], logger)

    # training for padding experiment
    padded_genuine_train = pad_states(os.path.join(args.genuine), max_len, logger)
    padded_bt_train = pad_states(os.path.join(args.bt), max_len, logger)

    padded_X_train, padded_y_train = create_dataset(padded_genuine_train, padded_bt_train)

    train_models(padded_X_train,
                 padded_y_train,
                 args.bt_name,
                 'padding',
                 out_dir,
                 logger)

    # training for averaging experiment
    averaged_genuine_train = average_states(os.path.join(args.genuine), logger)
    averaged_bt_train = average_states(os.path.join(args.bt), logger)

    averaged_X_train, averaged_y_train = create_dataset(averaged_genuine_train, averaged_bt_train)

    train_models(averaged_X_train,
                 averaged_y_train,
                 args.bt_name,
                 'averaging',
                 out_dir,
                 logger)


if __name__ == '__main__':
    args = parse_args()
    main(args)
