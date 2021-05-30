import numpy as np
import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray import tune

from SequentialSecondPrice import SequentialAuction

def get_rllib_config(seeds, debug=False, stop_iters=500):
    stop_config = {
        "training_iteration": 2 if debug else stop_iters,
    }
    env_config = {
        "nr_agents": 4
    }
    mock = SequentialAuction(env_config)
    rllib_config = {
        "env": SequentialAuction,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "DQN_policy": (None, mock.observation_space, mock.action_space, {})
            },
            "policy_mapping_fn": lambda agent_id: "DQN_policy",
        },
        "seed": tune.grid_search(seeds),
        "callbacks": {
            "on_episode_step": on_step
        },
        "num_gpus": 0,
        "framework": "tf2",
        "lr": 1e-5,
        "lr_schedule": [
            [0, 2.5e-4],
            [1e6, 1e-5]
        ],
        "train_batch_size": 128
    }

    return rllib_config, stop_config, env_config


def on_step(info):
    episode = info["episode"]
    nr_agents = 4
    state = np.zeros(shape=(2 + 2 * nr_agents,), dtype=np.int32)
    obs_p0 = episode.last_raw_obs_for(0)
    state[0:3] = obs_p0[0:3]
    state[2+nr_agents:] = obs_p0[2+nr_agents:]
    for x in range(1,nr_agents):
        state[2+x] = episode.last_raw_obs_for(x)[3]

    ssd = 0
    for x in range(nr_agents):
        ssd += (equilibrium_bid(x, state, nr_agents) - episode.last_action_for(x))**2

    episode.custom_metrics["ssd"] = ssd

def equilibrium_bid(player, state, nr_agents):
    has_won_bevore = state[2 + nr_agents + player] >= 1
    if has_won_bevore:
        return 0.0
    N = nr_agents
    K = state[0]
    k = state[1]
    x = state[2 + player]
    return x * (N - K) / (N - k)
def main():
    train_n_replicas = 1
    seeds = list(range(train_n_replicas))
    ray.init()
    rllib_config, stop_config, _ = get_rllib_config(seeds)
    tune_analysis = tune.run(DQNTrainer,
                             config=rllib_config,
                             stop=stop_config,
                             checkpoint_freq=20,
                             checkpoint_at_end=True,
                             name="Sequential",)
    ray.shutdown()

    return tune_analysis


if __name__ == "__main__":
    main()
