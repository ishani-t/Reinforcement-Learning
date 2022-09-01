import math

import gym
import numpy as np
import torch

import utils
from policies import QPolicy

# Modified by Mohit Goyal (mohit@illinois.edu) on 04/20/2022

class TabQPolicy(QPolicy):
    def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
        """
        Inititalize the tabular q policy

        @param env: the gym environment
        @param buckets: specifies the discretization of the continuous state space for each dimension
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate for the model update 
        @param gamma: discount factor
        @param model (optional): Load a saved table of Q-values for each state-action
            model = np.zeros(self.buckets + (actionsize,))
            
        """
        super().__init__(len(buckets), actionsize, lr, gamma)
        self.env = env
        self.buckets = buckets
        self.actionsize = actionsize
        self.lr = lr
        self.gamma = gamma

        # model is optional - either load existing values or make new one
        if(model is None):
            self.model = np.zeros(self.buckets + (actionsize,))
        else:
            self.model = model


    def discretize(self, obs):
        """
        Discretizes the continuous input observation

        @param obs: continuous observation
        @return: discretized observation  
        """
        upper_bounds = [self.env.observation_space.high[0], 5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def qvals(self, states):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        vals = []
        for state in states:
            s = self.discretize(state)
            q_val = self.model[s]
            vals.append(q_val)
        return vals


    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        
        updating_state = self.discretize(state) + (action,)
        curr_q_val = self.model[updating_state]
        
        if(done):
            target = reward
        else :
            # check action states - 0 and 1 for left and right
            left_val = self.model[self.discretize(next_state) + (0,)]
            right_val = self.model[self.discretize(next_state) + (1,)]
            max_val = max(left_val, right_val)
            target = reward + (self.gamma * max_val)
        
        alpha = self.lr
        updated_val = curr_q_val + alpha * (target - curr_q_val) 
        self.model[updating_state] = updated_val

        # return square error of the original q-value estimate and target
        loss = (target - curr_q_val ) ** 2
        return loss



    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    env.reset(seed=42) # seed the environment
    np.random.seed(42) # seed numpy
    import random
    random.seed(42)

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(1, 8, 1, 8), actionsize=actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'tabular.npy')
