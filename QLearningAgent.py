import numpy as np

import util
from agent import Agent


# TASK 3

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.3):
        """ A Q-Learning agent gets nothing about the mdp on construction other than a function mapping states to
        actions. The other parameters govern its exploration strategy and learning rate. """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction # this actionFunction just seem like is the getPossibleActions
                                            # why you dont just explain it
        self.qInitValue = 0  # initial value for states
        self.Q = {} # this will be a nasted dict 
        #like my_dict = {'key': {'value1': 10, 'value2': 20, 'value3': 30}}

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    def getValue(self, state):
        """ Look up the current value of the state. """
        # *********
        # TODO 3.1.
        # action_list =list(self.Q[state].keys())
        # Q_list = [self.getQValue(state,c_a)for c_a in action_list ]
        # return max(Q_list)
        return self.getQValue(state,self.getPolicy(state))
        # *********

    def getQValue(self, state, action):
        """ Look up the current q-value of the state action pair. """
        # *********
        # TODO 3.2.
        if state not in self.Q:
            return 0
        
        action_list =list(self.Q[state].keys())
        if action in action_list:
            return self.Q[state][action]
        else:
            return 0
        # *********

    def getPolicy(self, state):
        """ Look up the current recommendation for the state. """
        # *********
        # TODO 3.3.
        if state not in self.Q:
            return self.getRandomAction(state)
        else:
            action_list =list(self.Q[state].keys())
            Q_list = [self.getQValue(state,c_a)for c_a in action_list ]
            return action_list[Q_list.index(max(Q_list))]
        # *********

    def getRandomAction(self, state):
        all_actions = self.actionFunction(state)
        if len(all_actions) > 0:
            # *********
            return np.random.choice(all_actions)
            # *********
        else:
            return None

    def getAction(self, state):
        """ Choose an action: this will require that your agent balance exploration and exploitation as appropriate. """
        # *********
        # TODO 3.4.
        if np.random.rand() < self.epsilon:
            return self.getRandomAction(state)
        else:
            return self.getPolicy(state)
        # *********

    def update(self, state, action, nextState, reward):
        """ Update parameters in response to the observed transition. """
        # *********
        # TODO 3.5.
        if state not in self.Q:
            self.Q[state] = {action:0.0 for action in self.actionFunction(state)}
        else:
            self.Q[state][action] = (1-self.learningRate)*self.Q[state][action] + \
            self.learningRate * (reward+ self.discount * self.getValue(nextState))
        # *********
