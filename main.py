import random
from collections import deque
import os
import time
import torch

from unityagents import UnityEnvironment
import numpy as np
from Agent import Agent
from ReplayBuffer import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network


n_episodes = 2000
max_t = 1000
eps_start = 1.0
eps_end = 0.01
eps_decay = 0.995
seed=0

if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print("device : {}".format(device))
    env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    #Replay Buffer
    memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random.seed(seed), device)
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    agent = Agent(state_size,
                  action_size,
                  seed=seed,
                  lr=LR,
                  memory=memory,
                  update_every=UPDATE_EVERY,
                  batch_size=BATCH_SIZE,
                  gamma=GAMMA,
                  TAU=TAU ,
                  device=device)

    for i_episode in range(1, n_episodes + 1):
        #state = env.reset()
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            #next_state, reward, done, _ = env.step(action)
            env_info = env.step(action.astype(int))[brain_name]
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]
		
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 15.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            time.sleep(120)
            os.system("shutdown /s /t 1")
            break

