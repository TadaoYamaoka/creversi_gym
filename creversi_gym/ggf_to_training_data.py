import argparse
import os
import numpy as np

from creversi import *
from creversi import GGF

parser = argparse.ArgumentParser()
parser.add_argument('dir')
parser.add_argument('training_data')
parser.add_argument('--rating', type=int, default=2000)
parser.add_argument('--moves', type=int, default=45)
parser.add_argument('--max_num', type=int, default=100000000)
parser.add_argument('--draw_ratio', type=float, default=0.1)
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

win_count = { 'black': 0, 'white': 0, 'draw': 0 }
draw_ratio = args.draw_ratio
i = 0
for path in find_all_files(args.dir):
    print(path)
    ggf.parse_file(path)
    for ratings, moves, result in zip(ggf.ratings, ggf.moves, ggf.results):
        if ratings[0] < th_rating or ratings[1] < th_rating:
            continue
        if len(moves) < th_moves:
            continue

        if result == 0 and (win_count['black'] + win_count['white']) * draw_ratio < win_count['draw']:
            continue
        if result > 0:
            win_count['black'] += 1
        elif result < 0:
            win_count['white'] += 1
        else:
            win_count['draw'] += 1

        board.reset()
        for move in moves:
            board.to_bitboard(training_data[i]['bitboard'])
            training_data[i]['turn'] = board.turn
            training_data[i]['move'] = move
            if board.turn:
                reward = 1 if result > 0 else (-1 if result < 0 else 0)
            else:
                reward = -1 if result > 0 else (1 if result < 0 else 0)
            training_data[i]['reward'] = reward
            training_data[i]['done'] = False
            assert board.is_legal(move)
            board.move(move)
            i += 1

        training_data[i - 1]['done'] = True

    print(i)

training_data[:i].tofile(args.training_data)
print(win_count)
