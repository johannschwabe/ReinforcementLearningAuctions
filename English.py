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
        self.agents = list(range(env_config["nr_agents"]))
        self.nr_truthful_agents = env_config["nr_truthful_agents"]
        self.nr_players = len(self.agents) + self.nr_truthful_agents
        self.action_space = spaces.Discrete(self._nr_items+1)
        self.observation_space = spaces.Box(low=0, high=100*self._nr_items, shape=(1 + self.nr_players + self._nr_items,), dtype=np.int32)
        # self.observation_space = MultiAgentObservationSpace([
        #     spaces.Box(low=0, high=200, shape=(1,), dtype=np.int32),
        #     spaces.Discrete(3), spaces.Discrete(3),
        #     spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
        #     spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32)
        # ])
        self.reward_range = (-100 * self._nr_items, 100 * self._nr_items)
        self._state = np.zeros(shape=(1 + self.nr_players + self._nr_items * self.nr_players,), dtype=np.int32)
        self.reset()

    def reset(self):
        self._state = np.zeros(shape=(1 + self.nr_players + self._nr_items * self.nr_players,), dtype=np.int32)
        self._state[1:self.nr_players+1] = self._nr_items
        valuations = np.zeros((self.nr_players * self._nr_items,))
        for x in range(self.nr_players):
            valu = np.random.rand(1, self._nr_items)
            valuations[self._nr_items * x:self._nr_items * (x+1)] = -np.sort(-valu) * 100

        self._state[3:] = valuations

        obs_n = {}
        for i, agent in enumerate(self.agents):
            obs_n[i] = self._observation(i)
        return obs_n

    def step(self, action_n):
        # print("--------")
        # print(self._state)
        info = {}
        info[0] = self._state

        obs_n = {}
        done_n = {}

        for i, agent in enumerate(self.agents):
            obs_n[i] = self._observation(i)
        # print(obs_n)
        self._updateState(action_n)
        # print(self._state)

        done_n["__all__"] = np.sum(list(self._state[1:self.nr_players + 1])) <= self._nr_items or self._state[0] > 100 * self._nr_items
        reward_n = self._calculateRewards(done_n["__all__"])
        # print(reward_n)
        if not done_n["__all__"]:
            self._state[0] += 1
        return obs_n, reward_n, done_n, info

    def _calculateRewards(self, done):
        reward_n = {}
        for i, agent in enumerate(self.agents):
            if done:
                reward_n[i] = self._final_reward_i(i) / 100
            else:
                reward_n[i] = 0.0
        # print(type(reward_n[0]))
        return reward_n

    def _updateState(self, action_n):
        for player, bid in action_n.items():
            self._state[player + 1] = bid
        for ta in range(self.nr_truthful_agents):
            bid = 0
            for i in range(self._nr_items):
                if self._state[1 + self.nr_players + (len(self.agents) + ta) * self._nr_items + i] > self._state[0]:
                    bid += 1
            self._state[len(self.agents) + 1 + ta] = bid

    def _observation(self, i):
        my_index = i

        price = self._state[0]
        my_demand = self._state[my_index + 1]
        enemy_demand = []
        enemy_demand.extend(self._state[1:my_index])
        enemy_demand.extend(self._state[my_index+1:self.nr_players])
        my_valuations = self._state[1 + self.nr_players + my_index * self._nr_items: 1 + self.nr_players + self._nr_items * (my_index +1)]
        res = []
        res.extend([price, my_demand])
        res.extend(enemy_demand)
        res.extend(my_valuations)
        return res

    def _final_reward_i(self, i):
        bid = self._state[i + 1]
        if bid == 0:
            return 0

        my_value_index = 1 + self.nr_players + i * self._nr_items
        res = np.sum(self._state[my_value_index:my_value_index+bid])

        res -= self._state[0] * bid
        return res

    def render(self, mode='human'):
        pass



def create_english_env():
    env_config = {
        "agents": [0, 1]
    }
    return EnlishAuction(env_config)


tune.register_env("English", create_english_env)



