import numpy as np
import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray import tune

from English import EnlishAuction


def get_rllib_config(seeds, debug=False, stop_iters=300):
    stop_config = {
        "training_iteration": 2 if debug else stop_iters,
    }
    env_config = {
        "nr_agents": 1,
        "nr_items": 2,
        "nr_truthful_agents": 1
    }
    mock = EnlishAuction(env_config)
    rllib_config = {
        "env": EnlishAuction,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "DQN_policy": (None, mock.observation_space, mock.action_space, {})
            },
            "policy_mapping_fn": lambda agent_id: "DQN_policy",
        },
        "seed": tune.grid_search(seeds),
        "callbacks": {
            "on_episode_end": on_episode_end
        },
        "num_gpus": 0,
        "framework": "tf2",
        "eager_tracing": True,
        "lr": 1e-4,
        "train_batch_size": 128
    }

    return rllib_config, stop_config, env_config


def on_episode_end(info):
    episode = info["episode"]
    obs_p0 = episode.last_raw_obs_for(0)
    info_last_episode = episode.last_info_for(0)
    price = obs_p0[0]
    truthful_bid = 0
    truthful_price = np.sort(info_last_episode[3:])[1]
    if price < obs_p0[3]:
        truthful_bid += 1
        if price < obs_p0[4]:
            truthful_bid += 1
    episode.custom_metrics["dif_to_truthful"] = truthful_price - price
    episode.custom_metrics["truthful_bid"] = truthful_bid == obs_p0[1]
    episode.custom_metrics["overbid"] = truthful_bid < obs_p0[1]
    episode.custom_metrics["underbid"] = truthful_bid > obs_p0[1]
    episode.custom_metrics["price"] = price
    episode.custom_metrics["quantity"] = obs_p0[1]
    episode.custom_metrics["value_1"] = obs_p0[3]
    episode.custom_metrics["value_2"] = obs_p0[4]


def main():
    train_n_replicas = 1
    seeds = list(range(train_n_replicas))
    ray.init()
    rllib_config, stop_config, env_config = get_rllib_config(seeds)
    tune_analysis = tune.run(DQNTrainer,
                             config=rllib_config,
                             stop=stop_config,
                             checkpoint_freq=20,
                             checkpoint_at_end=True,
                             name="English",)
    ray.shutdown()

    return tune_analysis


main()
