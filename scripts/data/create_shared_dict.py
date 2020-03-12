#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Nicolas Spring

import argparse
from fairseq.tasks import FairseqTask

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+',
                        help='files for building the dict')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of concurrent workers')
    parser.add_argument('--threshold', type=int, default=-1,
                        help='defines the minimum word count')
    parser.add_argument('--nwords', type=int, default=-1,
                        help='defines the total number of words in the final dictionary, including special symbols')
    parser.add_argument('--padding-factor', type=int, default=8,
                        help='can be used to pad the dictionary size to be a multiple of 8, ' + \
                             'which is important on some hardware (e.g., Nvidia Tensor Cores)')
    parser.add_argument('--dict-out', type=str, required=True,
                        help='output path for the constructed dict')
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace):
    task = FairseqTask(None)
    dictionary = task.build_dictionary(filenames=args.filenames,
                                       workers=args.workers,
                                       threshold=args.threshold,
                                       nwords=args.nwords,
                                       padding_factor=args.padding_factor)
    dictionary.save(args.dict_out)

if __name__ == '__main__':
    args = parse_args()
    main(args)
