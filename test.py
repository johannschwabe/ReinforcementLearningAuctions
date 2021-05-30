import numpy as np
import ray

from English import EnlishAuction
from ray.rllib.agents.dqn import DQNTrainer
from train import get_rllib_config

checkpoint_path = "/Users/johannschwabe/ray_results/English/DQN_EnlishAuction_f6b0c_00000_0_seed=0_2021-05-28_22-12-01/checkpoint_000500/checkpoint-500"


def run():
    train_n_replicas = 1
    seeds = list(range(train_n_replicas))
    rllib_config, _, env_config = get_rllib_config(seeds)
    rllib_config["seed"] = 0
    rllib_config["explore"] = False
    ray.init()
    auction = EnlishAuction(env_config)
    player = DQNTrainer(config=rllib_config, env=EnlishAuction)
    player.restore(checkpoint_path=checkpoint_path,)
    steps = 200
    res = np.zeros((6,steps), dtype=np.float)
    for iteration in range(steps):
        # print("----- New Game -----")
        done = {"__all__": False}
        obs = auction.reset()
        reward = {}
        # print(obs)
        while not done["__all__"]:
            action_0 = player.compute_action(obs[0], policy_id="DQN_policy")
            action_1 = player.compute_action(obs[1], policy_id="DQN_policy")
            obs, reward, done, info = auction.step({0: action_0, 1: action_1})
            # print(f"A0: {action_0}, A1: {action_1}")
        obs_p0 = obs[0]
        price = obs_p0[0]
        res[:, iteration] = [price, price == 0, action_0 == 1 and action_1 == 1, action_0 + action_1 < 2, reward[0], reward[1]]
        # print(reward)
    return res
res = run()
print(np.mean(res, axis=1))