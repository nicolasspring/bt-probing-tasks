#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Nicolas Spring

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=argparse.FileType('r'),
                        help='input file (default: stdin)',
                        default=sys.stdin, metavar='PATH')
    args = parser.parse_args()
    return args


def normalized_distance_document(infile: '_io.TextIOWrapper') -> float:
    '''
    calculates normalized distance between corresponding words in a document.

    Args:
        infile  file containing alignments in i-j "Pharaoh format"

    Returns:
                the normalized distance between all alignments in the document
    '''
    n_sentence_pairs = 0
    distances_sum = 0
    for line in infile:
        distances_sum += normalized_distance_sentence(line)
        n_sentence_pairs += 1
    return distances_sum / n_sentence_pairs


def normalized_distance_sentence(line: str) -> float:
    '''
    calculates normalized distance between corresponding words in a sentence.

    Args:
        line    single line containing alignments in i-j "Pharaoh format"

    Returns:
                the normalized distance between all alignments in the sentence
    '''
    n_alignments = 0
    distances_sum = 0
    for pair in line.split():
        src, trg = pair.split('-')
        src, trg = int(src), int(trg)
        distance = src - trg
        # distance is always a positive number
        distances_sum += distance*-1 if distance < 0 else distance
        n_alignments += 1
    return distances_sum / n_alignments


def main(args: argparse.Namespace):
    print(f'normalized distance of alignments: {normalized_distance_document(args.input)}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
