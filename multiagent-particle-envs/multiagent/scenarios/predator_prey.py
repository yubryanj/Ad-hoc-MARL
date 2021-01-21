import numpy as np
from multiagent.core import World, Agent, Action
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, number_of_predators=4, number_of_preys=1):
        world = World()
        # add agents
        world.agents = [Agent() for i in range(number_of_predators + number_of_preys)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            
            if i < number_of_preys:
                agent.action_callback = self.scripted_agent_action
                agent.prey = True
            else:
                agent.action_callback = None
                agent.prey = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            if agent.prey:
                agent.color = np.array([1.0,0.00,0.0])    
            else:
                agent.color = np.array([0.00,1.0,0.00])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1,+1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def reward(self, agent, world):
        return -1

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + entity_pos)


    def scripted_agent_action(self, agent, world):
        # The scripted agent takes a random direction in every move
        
        scripted_agent_action = Action()
        action =  [[0, np.random.rand() * 2 - 1, 0, np.random.rand() * 2 - 1, 0]]
        
        scripted_agent_action.u = np.zeros(world.dim_p)
        scripted_agent_action.c = np.zeros(world.dim_c)

        scripted_agent_action.u[0] += action[0][1] - action[0][2]
        scripted_agent_action.u[1] += action[0][3] - action[0][4]

        sensitivity = 5.0
        if agent.accel is not None:
            sensitivity = agent.accel
        scripted_agent_action.u *= sensitivity

        return scripted_agent_action


    