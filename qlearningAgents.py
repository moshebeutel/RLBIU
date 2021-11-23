# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from os import stat
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
import numpy as np
from collections import defaultdict

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self._min_val = -1000.0
        self.Q = defaultdict(lambda:self._min_val)
        self.epsilon = float(args['epsilon'])
        self.alpha = float(args['alpha'])
        self.gamma = float(args['gamma'])
        self._step = 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.Q((state,action))
    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        # check if terminal state
        if not actions:
          return 0.0
        list_of_state_action_vals = [self.Q[(s,a)] for (s,a) in self.Q.keys() if s == state]
        return self._min_val if not list_of_state_action_vals else max(list_of_state_action_vals)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        # check if terminal state
        if not actions:
          return None
        best_val = self.computeValueFromQValues(state)
        list_of_best_actions = [a for (s,a) in self.Q.keys() if s == state and self.Q[(s,a)] == best_val]
        assert not list_of_best_actions or list_of_best_actions[0] in actions
        return None if not list_of_best_actions else random.choice(list_of_best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        "*** YOUR CODE HERE ***"
        self._step += 1
        if(self._step % 20 == 0 and self.Q.values()):
          print(f'step {self._step}, max Q value', max(list(self.Q.values())))
        actions = self.getLegalActions(state)
        # check if terminal state
        if not actions:
          print('max Q value', max(list(self.Q.values())))
          return None
        # epsilon greedy exploration
        if util.flipCoin(self.epsilon):
          return random.choice(actions)
        
        actionFromQValues = self.computeActionFromQValues(state)
         
        return actionFromQValues if actionFromQValues is not None else random.choice(actions)


        

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        next_state_max_val = self.computeValueFromQValues(nextState)
        self.Q[(state,action)] = self.Q[(state,action)] + self.alpha * (reward +  self.gamma*next_state_max_val - self.Q[(state,action)])
        # print(f'update state {state}, action {action}, nextState {nextState}, reward {reward}')
        # print('Q table')
        # print(self.Q)
        # print(max(list(self.Q.values())))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action
