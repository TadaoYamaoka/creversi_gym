import argparse

from creversi import *
from creversi_gym.player.random_player import RandomPlayer
from creversi_gym.player.softmax_player import SoftmaxPlayer

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='model.pt')
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--games', type=int, default=1)
parser.add_argument('--display', action='store_true')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    is_jupyter = get_ipython().__class__.__name__ != 'TerminalInteractiveShell'
    if is_jupyter:
        from IPython.display import SVG, display
except NameError:
    is_jupyter = False

players = [SoftmaxPlayer(args.model, device, args.temperature), RandomPlayer()]

black_won_count = 0
white_won_count = 0
draw_count = 0
board = Board()
for n in range(args.games):
    print(f'game {n}')
    board.reset()

    i = 0
    while not board.is_game_over():
        i += 1
        print(f'{i}: ' + ('black' if board.turn == BLACK_TURN else 'white'))
        if board.puttable_num() == 0:
            board.move_pass()
            print('pass')
            continue

        player = players[(i - 1) % 2]
        move = player.go(board)
        assert board.is_legal(move)
        if args.display:
            if is_jupyter:
                display(SVG(board.to_svg(move)))
            else:
                print(board)
        print(move_to_str(move))
        board.move(move)

    if args.display:
        if is_jupyter:
            display(SVG(board.to_svg(move)))
        else:
            print(board)

    if board.turn == BLACK_TURN:
        piece_nums = [board.piece_num(), board.opponent_piece_num()]
    else:
        piece_nums = [board.opponent_piece_num(), board.piece_num()]

    print(f'result black={piece_nums[0]} white={piece_nums[1]}')
    if piece_nums[0] > piece_nums[1]:
        print('black won')
        black_won_count += 1
    elif piece_nums[1] > piece_nums[0]:
        print('white won')
        white_won_count += 1
    else:
        print('draw')
        draw_count += 1

print(f'{black_won_count} {white_won_count} {draw_count}')