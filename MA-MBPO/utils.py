import argparse

def parse_args():
    parser = argparse.ArgumentParser("Ad-Hoc MARL")
    parser.add_argument("--scenario", type=str, default="simple_spread", help="Name of the MPE scenario")
    parser.add_argument("--maximum-episode-length", type=int, default=100, help="The length to run each episode")
    parser.add_argument("--time-steps", type=int, default=20000, help="Number of timesteps")
    parser.add_argument("--model-buffer-size", type=int, default=int(1e4), help="Size of the model buffer")
    parser.add_argument("--policy-buffer-size", type=int, default=int(1e4), help="Size of the poilcy buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="Size of each batch")
    parser.add_argument("--hidden_dimension", default=32, type=int, help='Hidden dimension size')
    parser.add_argument("--embedding_dimension", default=64, type=int, help='Default Embedding dimension')
    parser.add_argument("--action_embedding_dimension", default=64, type=int, help='Action Embedding dimension')
    parser.add_argument("--observation_embedding_dimension", default=64, type=int, help='Observation Embedding dimension')
    parser.add_argument("--state_embedding_dimension", default=64, type=int, help='State Embedding dimension')

    args = parser.parse_args()

    return args


def make_env():
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    args =  parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    # Turned shared viewer to false to have one window showing all the agents - does not work for interactive agent
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)

    args.action_dimension = env.world.dim_c
    args.observation_dimension = env.action_space[0].n

    return env, args
