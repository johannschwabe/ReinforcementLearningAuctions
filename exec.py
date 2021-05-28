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
        "agents": [0, 1],
        "nr_items": 3
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
        "framework": "tf",
        "lr": 5e-3,
        "train_batch_size": 128
    }

    return rllib_config, stop_config, env_config

def on_episode_end(info):
    episode = info["episode"]
    obs_p1 = episode.last_raw_obs_for(1)
    obs_p0 = episode.last_raw_obs_for(0)
    price = obs_p0[0]
    value_p0 = obs_p0[3:6]
    value_p1 = obs_p1[3:6]
    action_p0 = episode.last_action_for(0)
    action_p1 = episode.last_action_for(1)
    episode.custom_metrics["p=0"] = price == 0

    all_bids = []
    all_bids.extend(value_p0)
    all_bids.extend(value_p1)

    efficient_price = np.sort(all_bids)[2]

    truthful_q_p0 = 0
    i = 0
    while value_p0[i] > efficient_price:
        i += 1
        truthful_q_p0 += 1

    truthful_q_p1 = 0
    i = 0
    while value_p1[i] > efficient_price:
        i += 1
        truthful_q_p1 += 1

    episode.custom_metrics["inefficient"] = truthful_q_p0 != action_p0 or truthful_q_p1 != action_p1
    episode.custom_metrics["2-1"] = action_p0 == 2 and action_p1 == 1
    episode.custom_metrics["1-2"] = action_p0 == 1 and action_p1 == 2
    episode.custom_metrics["3-0"] = action_p0 == 3 and action_p1 == 0
    episode.custom_metrics["0-3"] = action_p0 == 0 and action_p1 == 3
    episode.custom_metrics["1-1"] = action_p0 == 1 and action_p1 == 1
    episode.custom_metrics["price"] = price
    episode.custom_metrics["q==3"] = (action_p0 + action_p1) == 3
    episode.custom_metrics["q==2"] = (action_p0 + action_p1) == 2
    episode.custom_metrics["q==1"] = (action_p0 + action_p1) == 1
    episode.custom_metrics["q==0"] = (action_p0 + action_p1) == 0

def main():
    train_n_replicas = 4
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