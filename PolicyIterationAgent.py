import numpy as np
from agent import Agent


# TASK 1

class PolicyIterationAgent(Agent):

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your policy iteration agent take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations

        states = self.mdp.getStates()
        number_states = len(states)
        # Policy initialization
        # ******************
        # TODO 1.1.a)
        self.V = {s: 0.0 for s in states} 
    
        
        # *******************

        self.pi = {s: self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states}

        counter = 0

        while True:
            # Policy evaluation
            for i in range(iterations):
                newV = {} # i need to do it in-place or not?
                for s in states:
                    a = self.pi[s]
                    # *****************
                    # TODO 1.1.b)
                    if mdp.isTerminal(s):
                        self.V[s] = 0.0
                    
                    else:
                        # self.pi[s] return the best action of state s
                        c_action= self.pi[s]
                        self.V[s] = sum([probs * (self.mdp.getReward(s,c_action,None) + self.discount* self.V[n_s])
                                    for n_s, probs in self.mdp.getTransitionStatesAndProbs(s,c_action)])
                    # i wasnt really sure is it can only have deterministic policy. but it should be

                # update value estimate
                # self.V=...

                # ******************

            policy_stable = True #what is this oh i got it
            for s in states:
                actions = self.mdp.getPossibleActions(s)
                if len(actions) < 1:
                    self.pi[s] = None
                else:
                    old_action = self.pi[s]
                    # ************
                    # TODO 1.1.c)
                    pi_list = [sum([probs * (self.mdp.getReward(s,c_a,None) + self.discount* self.V[n_s])
                                for n_s, probs in self.mdp.getTransitionStatesAndProbs(s,c_a)]) 
                                for c_a in actions]
                    self.pi[s] = actions[pi_list.index(max(pi_list))]

                    policy_stable = old_action==self.pi[s] and policy_stable

                    # ****************
            counter += 1

            if policy_stable: break

        print("Policy converged after %i iterations of policy iteration" % counter)

    def getValue(self, state):
        """
        Look up the value of the state (after the policy converged).
        """
        # *******
        # TODO 1.2.
        return  self.V[state]
        # ********

    def getQValue(self, state, action):
        """
        Look up the q-value of the state action pair
        (after the indicated number of value iteration
        passes).  Note that policy iteration does not
        necessarily create this quantity and you may have
        to derive it on the fly.
        """
        # *********
        # TODO 1.3.
        return sum([probs * (self.mdp.getReward(state,action,None) + self.discount* self.V[n_s])
                    for n_s, probs in self.mdp.getTransitionStatesAndProbs(state,action)])
        # **********

    def getPolicy(self, state):
        """
        Look up the policy's recommendation for the state
        (after the indicated number of value iteration passes).
        """
        # **********
        # TODO 1.4.
        return self.pi[state]
        # **********

    def getAction(self, state):
        """
        Return the action recommended by the policy.
        """
        return self.getPolicy(state)

    def update(self, state, action, nextState, reward):
        """
        Not used for policy iteration agents!
        """

        pass
