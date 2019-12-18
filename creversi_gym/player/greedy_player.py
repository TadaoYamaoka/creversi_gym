import numpy as np
import torch
import torch.nn.functional as F

class GreedyPlayer:
    def __init__(self, model_path, device, network='dqn'):
        if network == 'dueling':
            from creversi_gym.network.cnn10_dueling import DQN
        else:
            #from creversi_gym.network.cnn5 import DQN
            from creversi_gym.network.cnn10 import DQN
        self.device = device
        self.model = DQN().to(device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.features = np.empty((1, 2, 8, 8), np.float32)

    def go(self, board):
        with torch.no_grad():
            board.piece_planes(self.features[0])
            state = torch.from_numpy(self.features).to(self.device)
            q = self.model(state)
            # 合法手に絞る
            legal_moves = list(board.legal_moves)
            next_actions = torch.tensor([legal_moves], device=self.device, dtype=torch.long)
            legal_q = q.gather(1, next_actions)
            return legal_moves[legal_q.argmax(dim=1).item()]
