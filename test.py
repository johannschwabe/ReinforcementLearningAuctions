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
    ray.init()
    auction = EnlishAuction(env_config)
    player = DQNTrainer(rllib_config, EnlishAuction)
    player.restore(checkpoint_path)

    done = False
    obs = auction.reset()
    while not done:
        action = player.compute_action(obs, policy_id="DQN_policy")
        obs, reward, done, info = auction.step(action)
        print(action)
        print(reward)


run()
