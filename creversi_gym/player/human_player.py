from creversi import *
import random

RESIGN = -1
QUIT = -2

class HumanPlayer:
    def go(self, board):
        legal_moves = board.legal_moves
        if len(legal_moves) == 0:
            return PASS
        else:
            while True:
                move_str = input()
                if move_str == 'resign':
                    return RESIGN
                elif move_str == 'quit':
                    return QUIT
                try:
                    move = move_from_str(move_str)
                except:
                    print('invalid string')
                    continue
                if board.is_legal(move):
                    return move

