import numpy as np
import ray

from SequentialSecondPrice import SequentialAuction
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.ppo import PPOTrainer
from train_seq import get_rllib_config

# checkpoint_path = "/Users/johannschwabe/ray_results/Sequential/DQN_SequentialAuction_91812_00000_0_seed=0_2021-05-30_22-06-49/checkpoint_001500/checkpoint-1500"
checkpoint_path = "/Users/johannschwabe/ray_results/Sequential/PPO_SequentialAuction_46170_00001_1_seed=1_2021-05-31_08-34-38/checkpoint_001500/checkpoint-1500"
# checkpoint_path = "/Users/johannschwabe/ray_results/Sequential/DQN_SequentialAuction_cdb60_00003_3_seed=3_2021-05-30_22-08-30/checkpoint_001500/checkpoint-1500"


def equilibrium_bid(player, state, nr_agents):
    has_won_bevore = state[2 + nr_agents + player] >= 1
    if has_won_bevore:
        return 0.0
    N = nr_agents
    K = state[0]
    k = state[1]
    x = state[2 + player]
    return x * (N - K) / (N - k)

def eval():
    seeds =[0]
    rllib_config, _, env_config = get_rllib_config(seeds)
    rllib_config["seed"] = 0
    rllib_config["explore"] = False
    ray.init()
    player = PPOTrainer(config=rllib_config, env=SequentialAuction)
    player.restore(checkpoint_path=checkpoint_path, )
    sad = 0
    counter = 0
    nr_agents = 4
    has_one = 0
    for items in range(1,nr_agents):
        for value in range(0, 100, 5):
            print(f"---- value: {value} ----")
            for iter in range(1,items+1):
                counter += 1
                action = player.compute_action([items, iter, value, has_one], policy_id="PPO_policy")
                equilibrium_action = value * (nr_agents-items)/(nr_agents-iter)
                sad += abs(action-equilibrium_action)
                print(f"action: {action}, equ: {equilibrium_action}")
    print(f"avg. SAD: {sad/counter}")
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
        rounds = 0
        total_reward = 0
        reward = {}
        # print(obs)
        while not done["__all__"]:
            rounds += 1
            actions = {}
            for i in range(env_config["nr_agents"]):
                actions[i] = player.compute_action(obs[i], policy_id="PPO_policy")
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
                total_reward += reward[x]
                print(f"Value: {state[2 + x]}, Bid: {actions[x]}, equilibrium: {equilibrium_bid(x,state,nr_agents)}")
            print("...round...")
            obs = obs_new
        print(f"---game: {total_reward/rounds}---")
    print(sad/counter)
        # print(reward)
    return res


eval()
# res = run()
# print(np.mean(res, axis=1))