from utils import make_env, parse_args
from multiagent.policy import InteractivePolicy


def fit():
    env, args = make_env()
    env.render()
    
    agents = [InteractivePolicy(env,i) for i in range(env.n)]
    obs = env.reset()
    while True:
        actions = []
        for i, agent in enumerate(agents):
            actions.append(agent(obs[i]))
                
        obs_n, reward_n, done_n, _ = env.step(actions)

        env.render()

if __name__ == "__main__":
    fit()