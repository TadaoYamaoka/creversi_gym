import gym
import creversi.gym_reversi
from creversi import *

import os
import datetime
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('Reversi-v0').unwrapped

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

from network.cnn import DQN

def get_state(board):
    features = np.empty((1, 2, 8, 8), dtype=np.float32)
    board.piece_planes(features[0])
    state = torch.from_numpy(features[:1]).to(device)
    return state

######################################################################
# Training

BATCH_SIZE = 512
GAMMA = 0.7
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
OPTIMIZE_PER_EPISODES = 2
TARGET_UPDATE = 10

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=1e-5)
memory = ReplayMemory(10000)

def epsilon_greedy(state, legal_moves):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)

    score = 0
    if sample > eps_threshold:
        with torch.no_grad():
            q = policy_net(state)
            value, select = q[0, legal_moves].max(0)
    else:
        select = random.randrange(len(legal_moves))
    return select, value

temperature = 0.6
def softmax(state, legal_moves):
    with torch.no_grad():
        q = policy_net(state)
        log_prob = q[0, legal_moves] / temperature
        select = torch.distributions.categorical.Categorical(logits=log_prob).sample()
        value = q[0, legal_moves[select]]
    return select, value

steps_done = 0
def select_action(board, state):
    global steps_done

    steps_done += 1

    legal_moves = list(board.legal_moves)

    select, value = epsilon_greedy(state, legal_moves)
    #select, value = softmax(state, board.legal_moves)

    return legal_moves[select], torch.tensor([[legal_moves[select]]], device=device, dtype=torch.long), value, False


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
            non_final_next_actions_list.append(next_actions + [next_actions[0]] * (593 - len(next_actions)))
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
    target_q = target_net(non_final_next_states)
    next_state_values[non_final_mask] = target_q.gather(1, non_final_next_actions).max(1)[0].detach()
    # Compute the expected Q values
    # 相手番の価値のため反転する
    expected_state_action_values = (-next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    print(f"loss = {loss.item()}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################
# main training loop

# 棋譜保存用
os.makedirs('kifu', exist_ok=True)
kif = GGF.Exporter()

num_episodes = 1000
max_moves = 512
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    state = get_state(env)
    #env.render('sfen')
    kif.open(os.path.join('kifu', datetime.datetime.now().strftime('%Y%m%d%H%M%S') + '.kifu'))
    kif.header(['dqn', 'dqn'])
    
    for t in count():
        # Select and perform an action
        move, action, score, mate = select_action(env.board, state)
        reward, done, is_draw = env.step(move)

        #詰みの場合
        if mate:
            reward = 1.0
            done = True
        # 持将棋の場合
        if t + 1 == max_moves:
            done = True

        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = get_state(env)
            next_actions = get_legal_labels(env.board)
        else:
            next_state = None
            next_actions = None

        # 棋譜出力
        kif.move(move)
        kif.info('info score cp ' + str(score))
        if done:
            if is_draw == REPETITION_DRAW:
                kif.end('sennichite')
            elif is_draw == REPETITION_WIN:
                kif.end('illegal_win')
            elif is_draw == REPETITION_LOSE:
                kif.end('illegal_lose')
            elif t + 1 == max_moves:
                kif.end('draw')
            else:
                kif.end('resign')

        # Store the transition in memory
        memory.push(state, action, next_state, next_actions, reward)

        # Move to the next state
        state = next_state

        if done:
            kif.close()
            break

    if i_episode % OPTIMIZE_PER_EPISODES == OPTIMIZE_PER_EPISODES - 1:
        # Perform several episodes of the optimization (on the target network)
        optimize_model()

        # Update the target network, copying all weights and biases in DQN
        if i_episode // OPTIMIZE_PER_EPISODES % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.close()