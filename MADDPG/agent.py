import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon):

        # Epilson Greedy approach 

        # Epsilon
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        
        # Greedy
        else:
            # Convert the observation vector into a tensor
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)

            # Get the action from the actor given the observations
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))

            # Convert the action tensor into a np array
            u = pi.cpu().numpy()

            # Calculate the noise
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise

            # Apply noise to the action
            u += noise

            # Clip the action to within the range
            u = np.clip(u, -self.args.high_action, self.args.high_action)

        # Make a copy of the action space
        return u.copy()

    def learn(self, transitions, other_agents):

        # Train the agent
        self.policy.train(transitions, other_agents)

