import random

import numpy as np
import torch
from collections import deque, namedtuple


class ReplayBuffer:
    """
    Replay buffer to store named tuple.

    store and samples the (state, action, reward, next_state, done) tuples
    uses deque as datastructure to store those samples

    Attributes
    ----------
    buffer_size: int
    memory: deque
    batch_size: int
    experience: namedtuple
    device: str

    methods
    -------
    add(self, state, action, reward, next_state, done)
    sample()
    __len__()

    """
    def __init__(self, buffer_size, batch_size, seed, device):
        random.seed(seed)
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """ adds the state-action to the memory """
        exp_tuple = self.experience(state,action,reward,next_state,done)
        self.memory.append(exp_tuple)

    def sample(self):
        """ samples the tuples from the memory and stacks them vertically """

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """ overriding the default method for the length of memory"""
        return len(self.memory)