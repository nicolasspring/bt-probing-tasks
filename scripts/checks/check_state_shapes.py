#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Nicolas Spring

import argparse
import os
import torch

ENCODER_EMBED_DIM = 1024
MAX_TOKENS = 3584

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('states_dir', nargs='+', type=str,
                        help='directories containing extracted model states')
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace):
    print(f'Encoder size: {ENCODER_EMBED_DIM}')
    print(f'Max tokens: {MAX_TOKENS}\n')
    for directory in args.states_dir:
        print(f'Testing {directory}...')
        state_names = []
        for dirpath, _, names in os.walk(os.path.abspath(directory)):
            for name in names:
                if name.startswith('batch') and name.endswith('.pt'):
                    state_names.append(os.path.join(dirpath, name))

        states = (torch.load(name, map_location=torch.device('cpu')) for name in state_names)

        # shape of extracted states is T x B x C
        max_timesteps = 0
        n_states = 0
        for state in states:
            n_states += 1
            assert state.shape[2] == ENCODER_EMBED_DIM
            assert state.shape[0] * state.shape[1] <= MAX_TOKENS
            if state.shape[0] > max_timesteps:
                max_timesteps = state.shape[0]

        print('Test completed:')
        print(f'\t- Tested {n_states} model states.')
        print(f'\t- All states have encoder size {ENCODER_EMBED_DIM}.')
        print(f'\t- No states have more tokens than {MAX_TOKENS}.')
        print(f'\t- Highest number of time steps is {max_timesteps}.\n')

    print('All tests completed.')



if __name__ == '__main__':
    args = parse_args()
    main(args)
