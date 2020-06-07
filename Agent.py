import random
import torch
import numpy as np
import torch.optim as optim
from Model import Model

import torch.nn.functional as F




class Agent:
    """
    An Agent implememnted with vanilla DQNN. 

    Attributes
    ----------
        state_size : int
            state size of the environment 
        action_size : int
            action size of the environment  
        memory: ReplayBuffer
            Replay buffer for sampling stored state-action
        update_every: int
            learning interval, in steps
        batch_size: int
            Replay memory sampling batch size
        gamma: float
            discount rate
        TAU: float
            interpolation parameter
        device: str
            cpu/CUDA
        t_step: int
            time step
        optimizer: torch.optim
            optimizer for backpropagation 
        qnetwork_local: nn.Module
            neural network 
        qnetwork_target: nn.Module
        neural network

        
    Methods
    -------
        step(self, state, action, reward, next_state, done)
        learn(self, experiences, gamma)
        soft_update(self, local_model, target_model, tau)
        act(self, state, eps=0.)
        
    """

    def __init__(self, 
                state_size, 
                action_size, 
                seed, lr, 
                memory, 
                update_every, 
                batch_size, gamma, TAU, device,DDQN=False):
        """Constructor initializing Agent attributes 

        Params
        ======
            state_size(int)
            action_size(int)
            seed(int)
            lr(float)
            memory(ReplayBuffer)
            update_every(int)
            batch_size(int) 
            gamma(float) 
            TAU(float) 
            device(str)
        """

        random.seed(seed)
        self.DDQN=DDQN
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        self.device = device
        # Q-Network
        self.qnetwork_local = Model(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = Model(state_size, action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = memory
        self.t_step = 0

        self.update_every = update_every
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = TAU

    def step(self, state, action, reward, next_state, done):
        """Step to populate the replay buffer 

        Params
        ======
            state (tule): state
            action (tule): action
            reward (int): reward for state-action
            next_state (tuple): state
            done(int): 1 or 0 
        """
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) %  self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                exp = self.memory.sample()
                self.learn(exp)

    def learn(self, experiences):
        """learn backpropogation.

        td_target = reward + ( discount * max( q_target( state ) ) * ( 1 - done ) )

        Params
        ======
            experiences (tule): state-action tuples
        """
        states, actions, rewards, next_states, dones = experiences
        # Compute Q targets for next_states 
        Q_targets = self.DQN(rewards,next_states, dones) if not self.DDQN else self.doubleDQN(rewards,next_states, dones)

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def DQN(self, rewards, next_states, dones):
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for next_states 
        return rewards + ( self.gamma* Q_targets_next * (1 - dones))


    def doubleDQN(self, rewards, next_states, dones):
        # Get max predicted Q values (for next states) from target model
        Q_local_next_indices = self.qnetwork_local(next_states).detach().argmax(1).unsqueeze(1)
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,Q_local_next_indices)
        return rewards + ( self.gamma* Q_targets_next * (1 - dones))


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
