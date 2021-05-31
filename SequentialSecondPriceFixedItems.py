from gym import spaces
from ray import tune
from ray.rllib import MultiAgentEnv
import numpy as np

class SequentialAuctionFixedItems(MultiAgentEnv):
    def __init__(self, env_config):
        self._nr_agents = env_config["nr_agents"]
        self._nr_items = env_config["nr_items"]
        self._state = np.zeros(shape=(1+5*self._nr_agents))
        self.action_space = spaces.Discrete(100)
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.int32)
        self.reward_range = (-4, 1)
        self._state = np.zeros(shape=(1 + 2 * self._nr_agents), dtype=np.int32)
        self.reset()

    def reset(self):
        self._state = np.zeros(shape=(1 + 2 * self._nr_agents), dtype=np.int32)
        valuations = np.random.rand(1, self._nr_agents) * 100
        self._state[0] = 1
        self._state[1:1+self._nr_agents] = valuations
        obs_n = {}
        for i in range(self._nr_agents):
            obs_n[i] = self._observation(i)
        return obs_n


    def step(self, action_n):
        info = {}
        obs_n = {}
        done_n = {}

        done_n["__all__"] = self._nr_items == self._state[0]

        reward_n = self._updateState(action_n)


        for i in range(self._nr_agents):
            obs_n[i] = self._observation(i)
        return obs_n, reward_n, done_n, info

    def _updateState(self, action_n):
        bids = np.array(list(action_n.values()))
        winning_bid = np.max(bids)
        winning_players = np.where(bids == winning_bid)
        winning_player = np.random.choice(winning_players[0])
        second_highest_bid = -np.sort(-bids)[1]
        res = {}
        for player in range(self._nr_agents):
            if player != winning_player:
                res[player] = 0.0
            else:
                has_won_before = self._state[1 + self._nr_agents + winning_player] > 0
                valuation = self._state[1 + winning_player]

                if has_won_before:
                    valuation = 0

                res[player] = (valuation - second_highest_bid) / 100
                self._state[1 + self._nr_agents + winning_player] += 1
        self._state[0] += 1
        return res


    def _observation(self, i):
        obs = np.zeros(3)
        obs[0] = self._state[0]       #iteration
        obs[1] = self._state[1 + i]     #pi valuation
        obs[2] = self._state[1 + self._nr_agents + i]
        return obs


def create_seq_env():
    env_config = {
        "nr_agents": 4
    }
    return SequentialAuctionFixedItems(env_config)

if __name__ == "__main__":
    tune.register_env("Sequential", create_seq_env)