import numpy as np

class QAgent:

    def __init__(self, n_obs, n_act):
        self.n_obs = n_obs
        self.n_act = n_act

        self.Q_table = np.zeros((self.n_obs, self.n_act))
        self.exploration_proba = 0.1
        self.exploration_decreasing_decay = 0.0025 #0.005
        self.min_exploration_proba = 0.01
        self.gamma = 0.99
        self.lr = 0.1

    def take_action(self, s):

        action = None

        if np.random.uniform(0,1) < self.exploration_proba:
            action = np.random.choice([0,1,2,3], 1).item()
        else:
            action = np.argmax(self.Q_table[s,:])
        
        return action

    def improve_q_table(self, current_state, action, next_state, reward):

        self.Q_table[current_state, action] = self.Q_table[current_state, action] +self.lr*(reward + self.gamma*max(self.Q_table[next_state,:]) - self.Q_table[current_state, action])
    
    def reduce_exploration_rate(self, num_ep):
        
        self.exploration_proba = self.exploration_proba