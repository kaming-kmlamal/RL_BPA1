from agent import Agent
import numpy as np

# TASK 2
class ValueIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # *************
        #  TODO 2.1 a)
        # self.V = ...
        self.V = {s: 0.0 for s in states} 
        
        # ************

        for i in range(iterations):
            newV = {}  #for not in-place 
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                # **************
                # TODO 2.1. b)
                if mdp.isTerminal(s):
                    newV[s] = 0.0

                else:
                    a_list=[sum([probs * (self.mdp.getReward(s,c_a,None) + self.discount* self.V[n_s])
                                for n_s, probs in self.mdp.getTransitionStatesAndProbs(s,c_a)]) 
                                for c_a in actions]
                    newV[s] = max(a_list)
                # Update value function with new estimate
            self.V = newV

                # ***************

    def getValue(self, state):
        """
        Look up the value of the state (after the indicated
        number of value iteration passes).
        """
        # **********
        # TODO 2.2
        return  self.V[state]
        # **********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that value iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # ***********
        # TODO 2.3.
        return sum([probs * (self.mdp.getReward(state,action,None) + self.discount* self.V[n_s])
                    for n_s, probs in self.mdp.getTransitionStatesAndProbs(state,action)])
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """

        actions = self.mdp.getPossibleActions(state)
        if len(actions) < 1:
            return None

        else:

        # **********
        # TODO 2.4
            s= state
            pi_list = [sum([probs * (self.mdp.getReward(s,c_a,None) + self.discount* self.V[n_s])
            for n_s, probs in self.mdp.getTransitionStatesAndProbs(s,c_a)]) 
            for c_a in actions]
            return actions[pi_list.index(max(pi_list))]
        # ***********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for value iteration agents!
        """

        pass
