#!/usr/bin/env python
import argparse

from multiagent.environment import MultiAgentEnv
from multiagent.policy import RandomPolicy, InteractivePolicy
import multiagent.scenarios as scenarios
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='data_generate.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    dataset = []
    n_samples = int(1e5)
    reset_interval = 1000

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)

    # create interactive policies for each agent
    policies = [RandomPolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_old = env.reset()

    for iteration in tqdm(range(n_samples)):

        if iteration % reset_interval == 0:
            obs_old = env.reset()

        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_old[i]))
        
        # step environment
        obs_new, reward_new, done_new, _ = env.step(act_n)

        # Save the data
        dataset.append([obs_old, act_n, obs_new])

        # Set the old observation to the new observation
        obs_old = obs_new

    np.save('./dataset', dataset)

