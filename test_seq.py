import numpy as np
import ray

from SequentialSecondPrice import SequentialAuction
from ray.rllib.agents.dqn import DQNTrainer
from train_seq import get_rllib_config

checkpoint_path = "/Users/johannschwabe/ray_results/Sequential/DQN_SequentialAuction_91812_00000_0_seed=0_2021-05-30_22-06-49/checkpoint_000400/checkpoint-400"


def equilibrium_bid(player, state, nr_agents):
    has_won_bevore = state[2 + nr_agents + player] >= 1
    if has_won_bevore:
        return 0.0
    N = nr_agents
    K = state[0]
    k = state[1]
    x = state[2 + player]
    return x * (N - K) / (N - k)


def run():
    train_n_replicas = 1
    seeds = list(range(train_n_replicas))
    rllib_config, _, env_config = get_rllib_config(seeds)
    rllib_config["seed"] = 0
    rllib_config["explore"] = False
    ray.init()
    auction = SequentialAuction(env_config)
    player = DQNTrainer(config=rllib_config, env=SequentialAuction)
    player.restore(checkpoint_path=checkpoint_path,)
    steps = 10
    sad = 0
    counter = 0
    res = np.zeros((6, steps), dtype=np.float)
    for iteration in range(steps):
        # print("----- New Game -----")
        done = {"__all__": False}
        obs = auction.reset()
        reward = {}
        # print(obs)
        while not done["__all__"]:
            actions = {}
            for i in range(env_config["nr_agents"]):
                actions[i] = player.compute_action(obs[i], policy_id="DQN_policy")
            obs_new, reward, done, info = auction.step(actions)
            nr_agents = env_config["nr_agents"]
            state = np.zeros(shape=(2 + 2 * nr_agents,), dtype=np.int32)
            obs_p0 = obs[0]
            state[0:3] = obs_p0[0:3]
            for x in range(1, nr_agents):
                state[2 + x] = obs[x][2]
                state[2 + nr_agents + x] = obs[x][3]
            for x in range(nr_agents):
                sad += abs(actions[x] - equilibrium_bid(x,state,nr_agents))
                counter += 1
                print(f"Bid: {actions[x]}, equilibrium: {equilibrium_bid(x,state,nr_agents)}")
            print("...round...")
            obs = obs_new
        print("---game---")
    print(sad/counter)
        # print(reward)
    return res
res = run()
print(np.mean(res, axis=1))