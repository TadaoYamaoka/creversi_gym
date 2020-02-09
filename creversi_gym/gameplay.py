from creversi import *
from creversi_gym.player.random_player import RandomPlayer
from creversi_gym.player.greedy_player import GreedyPlayer
from creversi_gym.player.softmax_player import SoftmaxPlayer
from creversi_gym.player.human_player import HumanPlayer, RESIGN, QUIT

import torch

def main(player1, player2, model1='model.pt', model2=None, network1='dqn', network2='dqn', temperature=0.1, games=1, is_display=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_display:
        try:
            is_jupyter = get_ipython().__class__.__name__ != 'TerminalInteractiveShell'
            if is_jupyter:
                from IPython.display import SVG, display
        except NameError:
            is_jupyter = False

    players = []
    for player, model, network in zip([player1, player2], [model1, model2], [network1, network2]):
        if player == 'random':
            players.append(RandomPlayer())
        elif player == 'greedy':
            players.append(GreedyPlayer(model, device, network))
        elif player == 'softmax':
            players.append(SoftmaxPlayer(model, device, temperature, network))
        elif player == 'human':
            players.append(HumanPlayer())
        else:
            raise RuntimeError(f'{player} not found')

    black_won_count = 0
    white_won_count = 0
    draw_count = 0
    board = Board()
    for n in range(games):
        print(f'game {n}')
        board.reset()
        move = None

        i = 0
        while not board.is_game_over():
            i += 1

            if is_display:
                print(f'{i}: ' + ('black' if board.turn == BLACK_TURN else 'white'))
                if is_jupyter:
                    display(SVG(board.to_svg(move)))
                else:
                    print(board)

            if board.puttable_num() == 0:
                move = PASS
            else:
                player = players[(i - 1) % 2]
                move = player.go(board)
                if isinstance(player, HumanPlayer):
                    if move == RESIGN:
                        break
                    elif move == QUIT:
                        return
                assert board.is_legal(move)

            if is_display:
                print(move_to_str(move))

            board.move(move)

        if isinstance(player, HumanPlayer) and move == RESIGN:
            if board.turn == BLACK_TURN:
                print('white won')
                white_won_count += 1
            else:
                print('black won')
                black_won_count += 1
            continue

        if is_display:
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

    print(f'black:{black_won_count} white:{white_won_count} draw:{draw_count}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', choices=['random', 'greedy', 'softmax', 'human'])
    parser.add_argument('player2', choices=['random', 'greedy', 'softmax', 'human'])
    parser.add_argument('--model1', default='model.pt')
    parser.add_argument('--model2')
    parser.add_argument('--network1', default='dqn')
    parser.add_argument('--network2', default='dqn')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--games', type=int, default=1)
    parser.add_argument('--is_display', action='store_true')

    args = parser.parse_args()

    main(args.player1, args.player2, args.model1, args.model2, args.network1, args.network2, args.temperature, args.games, args.is_display)