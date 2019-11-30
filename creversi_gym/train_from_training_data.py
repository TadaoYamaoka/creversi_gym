import argparse
import os
import numpy as np
from collections import namedtuple
import random

from creversi import *
from creversi import GGF

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('training_data')
parser.add_argument('--model', default='model.pt')
parser.add_argument('--resume')
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--iterations', type=int, default=10000)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Replay Memory

class ReplayMemory(object):

    def __init__(self, training_data):
        self.training_data = training_data

    def sample(self, batch_size):
        return np.random.choice(self.training_data, batch_size, replace=False)

    def __len__(self):
        return len(self.training_data)

######################################################################
# DQN

from network.cnn import DQN

######################################################################
# Training

training_data = np.fromfile(args.training_data, TrainingData)

memory = ReplayMemory(training_data)


BATCH_SIZE = args.batchsize

target_net = DQN().to(device)

optimizer = optim.RMSprop(target_net.parameters(), lr=1e-5)
#optimizer = optim.SGD(target_net.parameters(), lr=0.01, momentum=0.9)

if args.resume:
    print('resume {}'.format(args.resume))
    checkpoint = torch.load(args.resume)
    target_net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


######################################################################
# Training loop

board = Board()
features = np.empty((BATCH_SIZE, 2, 8, 8), np.float32)

loss_sum = 0
log_interval = 100
for batch_idx in range(args.iterations):
    transitions = memory.sample(BATCH_SIZE)

    moves = []
    rewards = []
    for i, data in enumerate(transitions):
        board.set_bitboard(data['bitboard'], data['turn'])
        assert board.is_legal(data['move'])
        n = random.randint(0, 1)
        if n == 0:
            board.piece_planes(features[i])
            moves.append(data['move'])
        elif n == 1:
            board.piece_planes_rotate180(features[i])
            moves.append(move_rotate180(data['move']))

    state_batch = torch.from_numpy(features).to(device)
    action_batch = torch.tensor(moves, device=device, dtype=torch.long).view(-1, 1)
    reward_batch = torch.tensor(transitions['reward'], device=device, dtype=torch.float32).view(-1, 1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = target_net(state_batch).gather(1, action_batch)

    expected_state_action_values = reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in target_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    loss_sum += loss.item()
    if (batch_idx + 1) % log_interval == 0:
        print(f"loss = {loss_sum / log_interval}")
        loss_sum = 0

print('save {}'.format(args.model))
torch.save({'state_dict': target_net.state_dict(), 'optimizer': optimizer.state_dict()}, args.model)

print('Complete')