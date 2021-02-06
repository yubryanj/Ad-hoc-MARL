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

        """
        TODO: Select a random initialization of 1,2,3,5
        TODO: Test on 4
        """
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

    def reward(self, agent, world, \
                distance_threshold=0.1,\
                minimum_number_of_predators=2):
        """
        TODO: Write me!
        """

        # Position of the prey
        prey_position = world.scripted_agents[0].state.p_pos

        # Distance of the agent from the prey
        agent_distance = np.sum(np.square(agent.state.p_pos - prey_position))

        # # distance of other predators from the prey
        # predator_distances = [np.sum(np.square(predator.state.p_pos - prey_position)) for predator in world.policy_agents if predator.name is not agent.name]

        # predators_in_range = [True for predator_distance in predator_distances if predator_distance < distance_threshold]

        # # If the agent is in range and enough other predators are in range of the prey
        # if agent_distance < distance_threshold and np.sum(predators_in_range) >= minimum_number_of_predators - 1:
        #     return 0
        # else:
        #     return -(np.sum(predator_distances) + agent_distance)

        return -agent_distance


    def observation(self, agent, world):
        """
        TODO: Write me!
        """
        # get positions of all entities in this agent's reference frame
        agent_pos = []
        for scripted_agent in world.scripted_agents:
            agent_pos.append(scripted_agent.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + agent_pos)


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


    