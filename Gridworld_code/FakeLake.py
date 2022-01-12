import numpy as np 
from Dynamics_model import Dynamics_model
from frozen_lake import FrozenLakeEnv
from torch import optim
import os
import torch
from sklearn.metrics.pairwise import cosine_similarity

class FakeLake:
    def __init__ (self, size = 4, cmap = None):
        
        self.env = FrozenLakeEnv(cmap, None, True, size, 0.8)
        self.obs_dim = size*size
        self.act_dim = 1
        self.hidden_size = 16
        self.ens_size = 3
        self.dynamics = [Dynamics_model(self.obs_dim, self.act_dim, self.hidden_size).double() for i in range(self.ens_size)]

        self.dynamics_optim = [optim.Adam(params = self.dynamics[i].parameters(), lr = 0.01) for i in range(self.ens_size)]

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_state_target = []

        self.buffer_size = 1000
        self.prev_obs = None
        self.tot_eps = 0
        self.tot_samples = 0
        self.all_rews = []
        self.hlist = self.env.hlist
        
        mydir = os.path.basename(os.getcwd())
        self.file = open("results_"+ mydir + ".csv","w+")
        self.file.write('samples,reward,reward_model,similarity\n')

    def eval_agent(self, agent):

        solved = 0
        for i in range(300):

            s = self.env.reset()
            done = False

            for t in range(100):
                action = np.argmax(agent.Q_table[s,:])

                s, r, done, _ = self.env.step(action)
                solved +=r

                if done:
                    break
        
        return(solved/300)
    
    def eval_agent_learned(self, agent, adv):

        solved = 0
        for i in range(300):

            s = self.reset()
            done = False
            idx = None

            for t in range(100):
                action = np.argmax(agent.Q_table[s,:])#agent.take_action(s)

                if adv != None:
                    idx = np.argmax(adv.Q_table[s,action,:])

                s, r, done, _ = self.step(action, idx)
                solved +=r

                if done:
                    break
        
        #print("The task has been solved {}/300".format(solved))
        return(solved/300)

    def reward (self, state):

        if state == self.obs_dim-1:
            return 1
        else:
            return 0

    def obs_to_vector(self, obs):

        vec = np.zeros(self.obs_dim)
        vec[obs] = 1
        return vec

    def reset(self):
        obs = self.env.reset()

        if len(self.buffer_states)>0 and (np.random.rand(1) < 1):
            obs = np.argmax(self.buffer_states[np.random.choice(range(len(self.buffer_states)), 1).item()])

        self.prev_obs = self.obs_to_vector(obs)

        return obs

    def step(self, action, model_idx):

        model_in = np.append(self.prev_obs, action)
        model_in = torch.tensor(model_in)

        softmax = torch.nn.Softmax(dim=0)

        avg = []

        # Check if we are using adversary
        if model_idx == None:

            # If we are not, take average of probabilities
            for i in range(self.ens_size):

                avg.append(softmax(self.dynamics[i](model_in)).detach().numpy())
        else:
            
            # If we are, use just the adverarially selected model
            avg.append(softmax(self.dynamics[model_idx](model_in)).detach().numpy())

        mean = torch.tensor(np.mean(avg, axis=0))

        dist = torch.distributions.Categorical(mean)
        next_state=dist.sample()

        next_state = int(next_state.item())

        self.prev_obs=self.obs_to_vector(next_state)

        done = next_state in self.hlist or next_state == self.obs_dim-1

        r = self.reward(next_state)

        return next_state, r, done, None
    
    def collect_samples (self, agent, adversary):

        i_collected = 1
        self.tot_samples += 1

        while i_collected < 100:

            obs = self.env.reset()
            vec_obs = self.obs_to_vector(obs)
            done = False
            t = 0
            tot_rew = 0
            eps_count = 0
            
            while not done and t<100:

                prev_obs = obs
                vec_prev_obs = np.copy(vec_obs)

                action = agent.take_action(prev_obs)

                self.buffer_states.append(vec_prev_obs)
                self.buffer_actions.append([action])

                obs, r, done, _ = self.env.step(action)
                vec_obs = self.obs_to_vector(obs)
                tot_rew += r
                self.tot_samples += 1

                self.buffer_state_target.append(obs)

                if len(self.buffer_states) > self.buffer_size:
                    self.buffer_states.pop(0)
                    self.buffer_actions.pop(0)
                    self.buffer_state_target.pop(0)
                
                i_collected += 1
                t = t+1

                #if self.tot_samples%400 == 0 and self.tot_samples != 0:

                 #   ret_true = self.eval_agent(agent)
                  #  ret_learn = self.eval_agent_learned(agent,adversary)
                   # sim = self.model_similarity()
                    #print("After {} samples: {:.3f} (true env) | {:.3f} (learned env) | {:.3f} (similarity)".format(self.tot_samples,ret_true, ret_learn, sim))

                    #self.file.write('{},{},{},{}\n'.format(self.tot_samples, ret_true, ret_learn,sim))
                    #self.file.flush()

                    #self.all_rews = []
                
                if i_collected == 100:
                    break
                    

            eps_count += 1

            agent.reduce_exploration_rate(self.tot_eps)
            self.tot_eps += 1
            self.all_rews.append(tot_rew)

            
        eps_count = 0
    
    def improve_model(self, agent, adversary):

        # The adversary is just used for evaluation purposes
        self.collect_samples(agent, adversary)

        model_in = np.hstack((self.buffer_states, self.buffer_actions))
        len_in = len(self.buffer_states[0]) + len(self.buffer_actions[0])
        labels = [[el] for el in self.buffer_state_target]

        all_data = np.hstack((model_in, labels))

        for idx in range(self.ens_size): # train each model

            np.random.shuffle(all_data) # shuffle the dataset

            for i in range(100): # epochs

                batches = torch.utils.data.BatchSampler(all_data, 256, False)

                for batch in batches:
                    batch = torch.tensor(batch)
                    model_in = batch[:,:len_in]

                    model_out = self.dynamics[idx](model_in)
                    criterion_model = torch.nn.CrossEntropyLoss()
                    self.dynamics_optim[idx].zero_grad()

                    target = batch[:,len_in:].reshape(len(batch)).long()
                    model_loss = criterion_model(model_out, target)
                    model_loss.backward()

                    self.dynamics_optim[idx].step()
            #print(model_loss.item())
    
    def model_similarity(self):
        tot_sim = []

        for s in range(16-1):

            if s not in self.env.hlist:

                for action in range(4):
                    true_prob = np.zeros(self.obs_dim)

                    # transitions t are in the form [p, next_s, r, done]
                    for t in self.env.P[s][action]:
                        true_prob[t[1]] = true_prob[t[1]] + t[0]

                    obs = self.obs_to_vector(s)

                    model_in = np.append(obs, action)
                    model_in = torch.tensor(model_in)
                    softmax = torch.nn.Softmax(dim=0)
                    
                    for idx, model in enumerate(self.dynamics):
                        out = model(model_in)
                        out = softmax(out).detach().numpy()

                        sim = cosine_similarity([true_prob], [out])

                        tot_sim.append(sim.item())
        return np.mean(tot_sim)