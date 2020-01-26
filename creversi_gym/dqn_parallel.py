import gym
from creversi.gym_reversi.envs import ReversiVecEnv
from creversi import *

import argparse
import os
import datetime
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--ddqn', action='store_true')
parser.add_argument('--dueling', action='store_true')
parser.add_argument('--model', default='model.pt')
parser.add_argument('--resume')
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--num_iterations', type=int, default=10000)
parser.add_argument('--log')
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=args.log, level=logging.DEBUG)

BATCH_SIZE = args.batchsize
vecenv = ReversiVecEnv(BATCH_SIZE)

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


######################################################################
# Replay Memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'next_actions', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# DQN

if args.dueling:
    from network.cnn10_dueling import DQN
else:
    from network.cnn10 import DQN

def get_states(envs):
    features_vec = np.zeros((BATCH_SIZE, 2, 8, 8), dtype=np.float32)
    for i, env in enumerate(envs):
        env.board.piece_planes(features_vec[i])
    return torch.from_numpy(features_vec).to(device)

######################################################################
# Training

GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 500
OPTIMIZE_PER_STEPS = (60 * 16 + BATCH_SIZE - 1) // BATCH_SIZE
TARGET_UPDATE = 4

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-5)

if args.resume:
    print('resume {}'.format(args.resume))
    checkpoint = torch.load(args.resume)
    target_net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

memory = ReplayMemory(131072)

def epsilon_greedy(q, legal_moves):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * iterations_done / EPS_DECAY)

    if sample > eps_threshold:
        _, select = q[legal_moves].max(0)
    else:
        select = random.randrange(len(legal_moves))
    return select

temperature = 0.5
def softmax(q, legal_moves):
    log_prob = q[legal_moves] / temperature
    select = torch.distributions.categorical.Categorical(logits=log_prob).sample()
    return select

def select_actions(states, envs):
    select_moves = []

    with torch.no_grad():
        q_vec = policy_net(states)

        for env, q in zip(envs, q_vec):
            board = env.board

            legal_moves = list(board.legal_moves)

            select = epsilon_greedy(q, legal_moves)
            #select = softmax(q, legal_moves)

            select_moves.append(legal_moves[select])

        return select_moves, torch.tensor(select_moves, device=device, dtype=torch.long).view(-1, 1)


######################################################################
# Training loop

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # 合法手のみ
    non_final_next_actions_list = []
    for next_actions in batch.next_actions:
        if next_actions is not None:
            non_final_next_actions_list.append(next_actions + [next_actions[0]] * (30 - len(next_actions)))
    non_final_next_actions = torch.tensor(non_final_next_actions_list, device=device, dtype=torch.long)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # 合法手のみの最大値
    if args.ddqn:
        max_a = policy_net(non_final_next_states).gather(1, non_final_next_actions).max(1)[1].unsqueeze(1)
        target_q = target_net(non_final_next_states).gather(1, non_final_next_actions)
        # 相手番の価値のため反転する
        next_state_values[non_final_mask] = -target_q.gather(1, max_a).squeeze().detach()
    else:
        target_q = target_net(non_final_next_states)
        # 相手番の価値のため反転する
        next_state_values[non_final_mask] = -target_q.gather(1, non_final_next_actions).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = next_state_values * GAMMA + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    logging.info(f"{iterations_done}: loss = {loss.item()}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################
# main training loop

num_steps = args.num_iterations * OPTIMIZE_PER_STEPS
iterations_done = 0
for steps in range(num_steps):
    # Initialize the environment and state
    states = get_states(vecenv.envs)

    # Select and perform an action
    moves, actions = select_actions(states, vecenv.envs)
    rewards, dones = vecenv.step(moves)

    next_states = get_states(vecenv.envs)
    for i, (env, state, action, reward, done, next_state) in enumerate(zip(vecenv.envs, states, actions, rewards, dones, next_states)):
        state.unsqueeze_(0)
        action.unsqueeze_(0)
        next_state.unsqueeze_(0)

        # Observe new state
        if not done:
            next_actions = list(env.board.legal_moves)
        else:
            next_state = None
            next_actions = None

        reward = torch.tensor([reward], device=device)

        # Store the transition in memory
        memory.push(state, action, next_state, next_actions, reward)

    # Move to the next state
    states = next_states

    if steps >= 59 and steps % OPTIMIZE_PER_STEPS == OPTIMIZE_PER_STEPS - 1:
        iterations_done += 1

        # Perform several episodes of the optimization (on the target network)
        optimize_model()

        # Update the target network, copying all weights and biases in DQN
        if steps // OPTIMIZE_PER_STEPS % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

print('save {}'.format(args.model))
torch.save({'state_dict': target_net.state_dict(), 'optimizer': optimizer.state_dict()}, args.model)

print('Complete')
