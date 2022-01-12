import numpy as np

class QAdversary:

    def __init__(self, n_obs, n_act, n_ensembles, epsilon):
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_ensembles = n_ensembles
        self.exploration_proba = epsilon

        # 3D Q table for the adversary
        self.Q_table = np.zeros((self.n_obs, self.n_act, n_ensembles))

        self.gamma = 0.99
        self.lr = 0.1

    def take_action(self, s, a):

        adv_action = None

        if np.random.uniform(0,1) < self.exploration_proba:
            adv_action = np.random.choice(range(self.n_ensembles), 1).item()
        else:
            adv_action = np.argmax(self.Q_table[s,a,:])
        
        return adv_action

    def improve_q_table(self, current_state, action, adv_action, next_state, next_action, reward):

        self.Q_table[current_state, action, adv_action] = (1-self.lr) * self.Q_table[current_state, action, adv_action] +self.lr*(reward + self.gamma*max(self.Q_table[next_state,next_action,:]))
    
    def reduce_exploration_rate(self, num_ep):
        
        self.exploration_proba = self.exploration_proba