import argparse
import os
import numpy as np

from creversi import *
from creversi import GGF

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('training_data')
parser.add_argument('--rating', type=int, default=0)
parser.add_argument('--moves', type=int, default=1)
parser.add_argument('--max_num', type=int, default=100000000)
args = parser.parse_args()

th_rating = args.rating
th_moves = args.moves
max_num = args.max_num

board = Board()
ggf = GGF.Parser()

training_data = np.empty(max_num, dtype=TrainingData)

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1] == '.ggf':
                yield os.path.join(root, file)

i = 0
for path in find_all_files(args.dir):
    print(path)
    ggf.parse_file(path)
    for ratings, moves, result in zip(ggf.ratings, ggf.moves, ggf.results):
        if ratings[0] < th_rating or ratings[1] < th_rating:
            continue
        if len(moves) < th_moves:
            continue

        board.reset()
        for move in moves:
            if move != PASS:
                board.to_bitboard(training_data[i]['bitboard'])
                training_data[i]['turn'] = board.turn
                training_data[i]['move'] = move
                if board.turn:
                    reward = 1 if result > 0 else (-1 if result < 0 else 0)
                else:
                    reward = -1 if result > 0 else (1 if result < 0 else 0)
                training_data[i]['reward'] = reward
                i += 1
                assert board.is_legal(move)
            board.move(move)

    print(i)

training_data[:i].tofile(args.training_data)
