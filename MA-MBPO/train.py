from utils import make_env
from policy.marl_mbpo import MARL_MBPO
from tqdm import tqdm
import torch
import os
import copy


def fit():
    env, args = make_env()
    env.render()

    # Assumes the agent shares the same model
    policy = MARL_MBPO(args)
    agents = [policy for i in range(env.n)]
    
    rewards = []

    for time_step in tqdm(range(args.time_steps)):

        if time_step % args.maximum_episode_length == 0 :
            observations = env.reset()

        # Make a copy; changed in environment transition
        initial_obs = copy.deepcopy(observations)
        
        actions = []
        for i, observation in enumerate(observations):
            actions.append(agents[i].action(observation))
                
        observations, rewards, done, _ = env.step(actions)
        
        # Make a copy; changed in environment transition
        next_obs = copy.deepcopy(observations)
        
        # Store into the buffer
        policy.model_buffer.store(initial_obs, actions, next_obs, rewards)

        env.render()

        if time_step > args.batch_size:
            policy.train()

if __name__ == "__main__":
    fit()