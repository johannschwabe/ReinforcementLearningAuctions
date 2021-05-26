import gym
import numpy as np
import ray
import tensorflow as tf
from gym import spaces
from ray import tune
from ray.rllib import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer


class EnlishAuction(MultiAgentEnv):
    def __init__(self, env_config):
        self.agents = env_config["agents"]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.int32)
        # self.action_space = MultiAgentActionSpace([spaces.Discrete(3) for _ in range(len(self.agents))])
        # self.observation_space = MultiAgentObservationSpace([spaces.Box(low=0, high=100, shape=(1, 5), dtype=np.int16) for _ in range(len(self.agents))])
        self.reward_range = (-200, 200)
        self._state = np.zeros(shape=(1, 7), dtype=np.int32)
        self.reset()

    def reset(self):
        self._state = np.zeros(shape=(7,), dtype=np.int32)
        self._state[1:3] = 2
        valuations = np.zeros((4,))
        for x in range(2):
            valu = np.random.rand(1,2)
            valuations[2*x] = np.amax(valu) * 100
            valuations[1+2*x] = np.amin(valu) * 100

        self._state[3:7] = valuations

        obs_n = {}
        for i, agent in enumerate(self.agents):
            obs_n[i] = self._observation(i)
        return obs_n

    def step(self, action_n):
        obs_n = {}
        reward_n = {}
        done_n = {}

        for i, agent in enumerate(self.agents):
            obs_n[i] = self._observation(i)

        self._updateState(action_n)

        done_n["__all__"] = np.sum(list(action_n.values())) <= 2

        for i, agent in enumerate(self.agents):
            if done_n["__all__"]:
                reward_n[i] = self._final_reward_i(i)
            else:
                reward_n[i] = 0

        if not done_n["__all__"]:
            self._state[0] += 1
        return obs_n, reward_n, done_n, {}

    def _updateState(self, action_n):
        for player, bid in action_n.items():
            self._state[player + 1] = bid

    def _observation(self, i):
        my_index = i
        enemy_index = (i + 1) % 2

        price = self._state[0]
        my_demand = self._state[my_index + 1]
        enemy_demand = self._state[enemy_index + 1]
        my_valuations = self._state[3 + my_index * 2: 5 + my_index * 2]
        res = []
        res.extend([price,my_demand,enemy_demand])
        res.extend(my_valuations)
        return res

    def _final_reward_i(self, i):
        bid = self._state[i+1]
        if bid == 0:
            return 0
        res = self._state[3 + i * 2]
        if bid == 2:
            res += self._state[4 + i * 2]
        res -= self._state[0] * bid
        return res


    def render(self, mode='human'):
        pass


def get_rllib_config(seeds, debug=False, stop_iters=200):
    stop_config = {
        "training_iteration": 2 if debug else stop_iters,
    }
    env_config = {
        "agents": [0, 1]
    }
    mock = EnlishAuction(env_config)
    rllib_config = {
        "env": EnlishAuction,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "ppo_policy": (None, mock.observation_space, mock.action_space, {})
            },
            "policy_mapping_fn": lambda agent_id: "ppo_policy",
        },
        "seed": tune.grid_search(seeds),
        "num_gpus": 0,
        "framework": "tf",
        "lr": 5e-3,
        "train_batch_size": 128
    }

    return rllib_config, stop_config, env_config


def main():
    train_n_replicas = 1
    seeds = list(range(train_n_replicas))
    ray.init()
    rllib_config, stop_config, env_config = get_rllib_config(seeds)
    tune_analysis = tune.run(PPOTrainer,
                             config=rllib_config,
                             stop=stop_config,
                             checkpoint_freq=0,
                             checkpoint_at_end=True,
                             name="PPO_English")
    ray.shutdown()
    return tune_analysis

main()