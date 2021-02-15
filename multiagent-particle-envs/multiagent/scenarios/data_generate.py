import numpy as np
from multiagent.core import World, Agent
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(4)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)


    def reward(self, agent, world):
        return 0

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        return np.concatenate([agent.state.p_vel] + entity_pos)
