import gym
import numpy as np
import ray
from gym import spaces
from ray import tune
from ray.rllib import MultiAgentEnv, rollout
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.dqn import DQNTrainer, DQNTFPolicy
from observation_space import MultiAgentObservationSpace
import tensorflow as tf


class EnlishAuction(MultiAgentEnv):
    def __init__(self, env_config):
        self._nr_items = env_config["nr_items"]
        self.agents = env_config["agents"]
        self.action_space = spaces.Discrete(self._nr_items+1)
        self.observation_space = spaces.Box(low=0, high=200, shape=(3+self._nr_items,), dtype=np.int32)
        # self.observation_space = MultiAgentObservationSpace([
        #     spaces.Box(low=0, high=200, shape=(1,), dtype=np.int32),
        #     spaces.Discrete(3), spaces.Discrete(3),
        #     spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
        #     spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)
        # ])
        self.reward_range = (-100 * self._nr_items, 100 * self._nr_items)
        self._state = np.zeros(shape=(3+self._nr_items*2,), dtype=np.int32)
        self.reset()

    def reset(self):
        self._state = np.zeros(shape=(3+self._nr_items*2,), dtype=np.int32)
        self._state[1:3] = self._nr_items
        valuations = np.zeros((2*self._nr_items,))
        for x in range(2):
            valu = np.random.rand(1, self._nr_items)
            valuations[self._nr_items * x:self._nr_items * (x+1)] = -np.sort(-valu) * 100

        self._state[3:] = valuations

        obs_n = {}
        for i, agent in enumerate(self.agents):
            obs_n[i] = self._observation(i)
        return obs_n

    def step(self, action_n):
        info = {}

        obs_n = {}
        done_n = {}
        # info["price"] = self._state[0]
        # info["bid_p0"] = self._state[1]
        # info["bid_p1"] = self._state[2]

        for i, agent in enumerate(self.agents):
            obs_n[i] = self._observation(i)
        self._updateState(action_n)

        done_n["__all__"] = np.sum(list(action_n.values())) <= self._nr_items or self._state[0] > 100 * self._nr_items
        reward_n = self._calculateRewards(done_n["__all__"])

        if not done_n["__all__"]:
            self._state[0] += 1
        return obs_n, reward_n, done_n, info

    def _calculateRewards(self, done):
        reward_n = {}
        for i, agent in enumerate(self.agents):
            if done:
                reward_n[i] = self._final_reward_i(i) / 100
            else:
                reward_n[i] = 0
        return reward_n

    def _updateState(self, action_n):
        for player, bid in action_n.items():
            self._state[player + 1] = bid

    def _observation(self, i):
        my_index = i
        enemy_index = (i + 1) % 2

        price = self._state[0]
        my_demand = self._state[my_index + 1]
        enemy_demand = self._state[enemy_index + 1]
        my_valuations = self._state[3 + my_index * 2: 3 + self._nr_items + my_index * 2]
        res = []
        res.extend([price, my_demand, enemy_demand])
        res.extend(my_valuations)
        return res

    def _final_reward_i(self, i):
        bid = self._state[i + 1]
        if bid == 0:
            return 0

        my_value_index = 3 + i * self._nr_items
        res = np.sum(self._state[my_value_index:my_value_index+bid])

        res -= self._state[0] * bid

        return res

    def render(self, mode='human'):
        print(self._state[0])



def create_english_env():
    env_config = {
        "agents": [0, 1]
    }
    return EnlishAuction(env_config)


tune.register_env("English", create_english_env)



