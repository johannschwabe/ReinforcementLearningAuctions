# from https://github.com/koulanurag/ma-gym

import gym


class MultiAgentActionSpace(gym.Space):

    def __init__(self, agents_action_space):
        for x in agents_action_space:
            assert isinstance(x, gym.spaces.space.Space)

        super(MultiAgentActionSpace, self).__init__(agents_action_space)
        self._agents_action_space = agents_action_space

    def sample(self):
        """ samples action for each agent from uniform distribution"""
        return [agent_action_space.sample() for agent_action_space in self._agents_action_space]

    def contains(self, obs):
        """ contains observation """
        for space, ob in zip(self._agents_action_space, obs):
            if not space.contains(ob):
                return False
        else:
            return True
