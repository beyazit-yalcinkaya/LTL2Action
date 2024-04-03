"""
This is a simple wrapper that will include DFA goals to any given environment.
It also progress the formulas as the agent interacts with the envirionment.

However, each environment must implement the followng functions:
    - *get_events(...)*: Returns the propositions that currently hold on the environment.
    - *get_propositions(...)*: Maps the objects in the environment to a set of
                            propositions that can be referred to in DFA.

Notes about DFAEnv:
    - The episode ends if the DFA goal is progressed to True or False.
    - If the DFA goal becomes True, then an extra +1 reward is given to the agent.
    - If the DFA goal becomes False, then an extra -1 reward is given to the agent.
    - Otherwise, the agent gets the same reward given by the original environment.
"""


import numpy as np
import gym
from gym import spaces
import random
from dfa_samplers import getDFASampler, SequenceSampler

class DFAEnv(gym.Wrapper):
    def __init__(self, env, progression_mode="full", dfa_sampler=None, intrinsic=0.0):
        """
        DFA environment
        --------------------
        It adds an DFA objective to the current environment
            - The observations become a dictionary with an added "text" field
              specifying the DFA objective
            - It also automatically progress the formula and generates an
              appropriate reward function
            - However, it does requires the user to define a labeling function
              and a set of training formulas
        progression_mode:
            - "full": the agent gets the full, progressed DFA formula as part of the observation
            - "partial": the agent sees which propositions (individually) will progress or falsify the formula
            - "none": the agent gets the full, original DFA formula as part of the observation
        """
        super().__init__(env)
        self.progression_mode = progression_mode
        self.propositions = self.env.get_propositions()
        self.sampler = getDFASampler(dfa_sampler, self.propositions)

        self.observation_space = spaces.Dict({'features': env.observation_space})
        self.intrinsic = intrinsic


    def sample_dfa_goal(self):
        # This function must return an DFA formula for the task
        # Format:
        #(
        #    'and',
        #    ('until','True', ('and', 'd', ('until','True',('not','c')))),
        #    ('until','True', ('and', 'a', ('until','True', ('and', 'b', ('until','True','c')))))
        #)
        # NOTE: The propositions must be represented by a char
        raise NotImplementedError

    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        raise NotImplementedError

    def reset(self):
        self.obs = self.env.reset()

        # Defining an DFA goal
        self.dfa_goal     = self.sample_dfa_goal()
        self.dfa_original = self.dfa_goal

        # Adding the DFA goal to the observation
        if self.progression_mode == "partial":
            dfa_obs = {'features': self.obs,'progress_info': self.progress_info(self.dfa_goal)}
        else:
            dfa_obs = {'features': self.obs,'text': self.dfa_goal}
        return dfa_obs

    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, env_done, info = self.env.step(action)

        # progressing the DFA formula
        truth_assignment = self.get_events(self.obs, action, next_obs)

        self.dfa_goal = self.progression(self.dfa_goal, truth_assignment)
        self.obs      = next_obs

        dfa_reward, dfa_done = self.get_dfa_reward_and_dfa_done(self.dfa_goal)

        # Computing the new observation and returning the outcome of this action
        if self.progression_mode == "full":
            dfa_obs = {'features': self.obs,'text': self.dfa_goal}
        elif self.progression_mode == "none":
            dfa_obs = {'features': self.obs,'text': self.dfa_original}
        elif self.progression_mode == "partial":
            dfa_obs = {'features': self.obs, 'progress_info': self.progress_info(self.dfa_goal)}
        else:
            raise NotImplementedError

        reward  = original_reward + dfa_reward
        done    = env_done or dfa_done
        return dfa_obs, reward, done, info

    def get_dfa_reward_and_dfa_done(self, dfa):
        start_state = dfa.start
        start_state_label = dfa._label(start_state)
        states = dfa.states()

        if start_state_label == True: # If starting state of self.dfa_goal is accepting, then dfa_reward is 1.0.
            dfa_reward = 1.0
            dfa_done = True
        elif len(states) == 1: # If starting state of self.dfa_goal is rejecting and self.dfa_goal has a single state, then dfa_reward is reject_reward.
            dfa_reward = -1 # Or maybe 0.0
            dfa_done = True
        else:
            dfa_reward = 0.0 # If starting state of self.dfa_goal is rejecting and self.dfa_goal has a multiple states, then dfa_reward is 0.0.
            dfa_done = False

        return dfa_reward, dfa_done

    def progression(self, dfa, truth_assignment, start=None):
        import attr
        dfa = attr.evolve(dfa, start=dfa.transition(truth_assignment, start=start))
        return dfa


    # # X is a vector where index i is 1 if prop i progresses the formula, -1 if it falsifies it, 0 otherwise.
    def progress_info(self, dfa):
        propositions = self.env.get_propositions()
        X = np.zeros(len(self.propositions))

        for i in range(len(propositions)):
            progress_i = self.progression(dfa, propositions[i])
            dfa_reward, _ = self.get_dfa_reward_and_dfa_done(progress_i)
            X[i] = dfa_reward
        return X

    def sample_dfa_goal(self):
        return self.sampler.sample()


    def get_events(self, obs, act, next_obs):
        # This function must return the events that currently hold on the environment
        # NOTE: The events are represented by a string containing the propositions with positive values only (e.g., "ac" means that only propositions 'a' and 'b' hold)
        return self.env.get_events()


class NoDFAWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Removes the DFA formula from an DFAEnv
        It is useful to check the performance of off-the-shelf agents
        """
        super().__init__(env)
        self.observation_space = env.observation_space
        # self.observation_space =  env.observation_space['features']

    def reset(self):
        obs = self.env.reset()
        # obs = obs['features']
        # obs = {'features': obs}
        return obs

    def step(self, action):
        # executing the action in the environment
        obs, reward, done, info = self.env.step(action)
        # obs = obs['features']
        # obs = {'features': obs}
        return obs, reward, done, info

    def get_propositions(self):
        return list([])
