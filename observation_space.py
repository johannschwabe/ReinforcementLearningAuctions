# from https://github.com/koulanurag/ma-gym
import gym
import numpy as np


class MultiAgentObservationSpace(gym.Space):
    def __init__(self, agents_observation_space):
        for x in agents_observation_space:
            assert isinstance(x, gym.spaces.space.Space)

        super().__init__(agents_observation_space)
        self._agents_observation_space = agents_observation_space
        self.shape = (len(agents_observation_space),)
        self.dtype = np.dtype(np.int32)

    def sample(self):
        """ samples observations for each agent from uniform distribution"""
        res = []
        for agent_observation_space in self._agents_observation_space:
            sample = agent_observation_space.sample()
            if hasattr(sample, '__iter__'):
                res.extend(sample)
            else:
                res.append(sample)
        return res

    def contains(self, obs):
        """ contains observation """
        for space, ob in zip(self._agents_observation_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True

