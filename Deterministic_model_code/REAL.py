from ast import Param
import gym
import numpy as np
from collections import deque
from random import randrange
import random

# Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal

# Networks
from networks.adversary_policy import Adversary_policy
from networks.critic import Critic
from networks.learned_model import Learned_model
from networks.learned_reward import Learned_reward
from networks.policy_net import Policy_net

# Utils
from utils import eval_policy

# Params
from params import Params

class REAL:
    
    def __init__(self):
        
        self._init_hyperparameters()
        self.writer = SummaryWriter()

        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu', index = 0)

        # Extract environment information
        self.env = gym.make(self.ENV_NAME)
        
        self.obs_dim = self.env.observation_space.shape[0]
        
        
        if self.env.action_space.__class__.__name__ == 'Discrete':
            self.act_dim = self.env.action_space.n
        else:
            self.act_dim = self.env.action_space.shape[0]
            self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5, device = self.dev)
            self.cov_mat = torch.diag(self.cov_var, device = self.dev)
                
        self.act_space = self.env.action_space.__class__.__name__
        
        if not self.MODEL_BASED:
            self.ADVERSARIAL = False
            self.timesteps_per_batch = 1000
            print("Since the algorithm is not model based, the adversary agent is not used.")
            print("Using 1k samples per iteration.")

        # Initialize model ensemble, reward, actor and critic networks
        
        self.model = [Learned_model(self.obs_dim, 1, self.MODEL_HIDDEN_SIZE, self.dev).to(self.dev) for i in range(self.ENSEMBLE_SIZE)]
        
        self.reward_net = Learned_reward(self.obs_dim, 1, self.REW_HIDDEN_SIZE, self.dev).to(self.dev)
        
        self.actor = Policy_net(observation_space_size=self.obs_dim,
                            action_space_size=self.act_dim,
                            hidden_size=self.POLICY_HIDDEN_SIZE, device = self.dev).to(self.dev)
        
        self.critic = Critic(self.obs_dim, 0, self.CRITIC_HIDDEN_SIZE, self.dev).to(self.dev)
        
        self.adversary = Adversary_policy(observation_space_size = self.obs_dim + 1,
                                         action_space_size = self.ENSEMBLE_SIZE,
                                         hidden_size = self.ADVERSARY_HIDDEN_SIZE, device = self.dev).to(self.dev)

        # Initialize optimizers for models, reward, actor, critic and adversary
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.ALPHA)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr= self.ALPHA)#self.lr)
        self.adam_model = [optim.Adam(params=self.model[i].parameters(), lr=self.ALPHA) for i in range(self.ENSEMBLE_SIZE)]
        self.adam_rew = optim.Adam(params=self.reward_net.parameters(), lr=self.ALPHA)
        self.adv_optim = optim.Adam(self.adversary.parameters(), lr = self.ALPHA)
        
        # Initialize dataset for model training
        self.dataset = torch.empty(size=(0,), dtype=torch.long, device = self.dev)
        self.dataset_labels = torch.empty(size=(0,), dtype=torch.long, device = self.dev)
        
        # Deque to keep track of the last 100 rewards (to check if game is solved)
        self.total_rewards = deque([], maxlen=100)
        
        self.rewards_hist = []
        
    def _init_hyperparameters(self):
        
        # Initialize default values for hyperparameters
        self.MODEL_BASED = Params.MODEL_BASED
        self.ADVERSARIAL = Params.ADVERSARIAL
        self.ALPHA = Params.ALPHA
        self.MODEL_BATCH_SIZE = Params.MODEL_BATCH_SIZE
        self.MODEL_EPOCHS = Params.MODEL_EPOCHS
        #self.N_ROLLOUTS = Params.N_ROLLOUTS
        self.DATASET_SIZE = Params.DATASET_SIZE
        self.GAMMA = Params.GAMMA
        self.POLICY_HIDDEN_SIZE = Params.POLICY_HIDDEN_SIZE
        self.MODEL_HIDDEN_SIZE = Params.MODEL_HIDDEN_SIZE
        self.ADVERSARY_HIDDEN_SIZE = Params.ADVERSARY_HIDDEN_SIZE
        self.REW_HIDDEN_SIZE = Params.REW_HIDDEN_SIZE
        self.ROLLOUT_LEN = Params.ROLLOUT_LEN
        self.ENSEMBLE_SIZE = Params.ENSEMBLE_SIZE
        self.CRITIC_HIDDEN_SIZE = Params.CRITIC_HIDDEN_SIZE
        self.EPSILON = Params.EPSILON
        self.ENV_NAME = Params.ENV_NAME
        self.MODEL_SAMPLES_ROLLOUT_LEN = Params.MODEL_SAMPLES_ROLLOUT_LEN
        self.SAMPLES_FOR_MODEL = Params.SAMPLES_FOR_MODEL
        self.TOTAL_ITERATIONS = Params.TOTAL_ITERATIONS
        
        
        # Algorithm hyperparameters
        self.timesteps_per_batch = Params.TIMESTEPS_PER_BATCH                 # Number of timesteps to run per batch
        self.n_updates_per_iteration = Params.N_UPDATES_PER_ITERATION                # Number of times to update actor/critic per iteration
        self.clip = Params.CLIP                                 # Recommended 0.2, helps define the threshold to clip the ratio during SGA
        
    def perform_rollouts (self):

        data_in = []
        data_out = []
        samples = 0

        self.actor.to(torch.device("cpu"))
        self.actor.dev = torch.device("cpu")
        
        while len(data_in) < self.SAMPLES_FOR_MODEL:#for i in range(n_rolls):

            obs = self.env.reset()

            for t in range(self.MODEL_SAMPLES_ROLLOUT_LEN):
                
                samples += 1
                action, _ = self.get_action(obs)

                sa_pair = torch.cat((torch.tensor(obs).float().unsqueeze(dim=0), action), 1)
                data_in.append(sa_pair)

                if self.act_space == 'Discrete':
                    action = int(action[0].cpu().item())
                else:
                    action = action.cpu().item()
                    if self.act_dim > 1:
                        action = action.detach().numpy()
                    else:
                        action = [action]
            
                obs_next, r, done, info = self.env.step(action)

                data_label = torch.cat((torch.tensor(obs_next).float().unsqueeze(dim=0), torch.tensor([r]).float().unsqueeze(dim=0)), 1)
                data_out.append(data_label)

                obs = obs_next

                if done or samples == self.SAMPLES_FOR_MODEL:
                    break

        data_in = torch.cat(data_in).to(self.dev)
        data_out = torch.cat(data_out).to(self.dev)

        self.actor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu', index = 0))
        self.actor.dev = torch.device(torch.device('cuda' if torch.cuda.is_available() else 'cpu', index = 0))

        return data_in, data_out

    def improve_model(self, t):
        
        loss_dict = {}

        # Perform rollouts with the current policy

        data, label = self.perform_rollouts()

        # Keep size of dataset <= self.DATASET_SIZE by removing the oldest transitions
        if self.dataset.size()[0] + data.size()[0] > self.DATASET_SIZE:
            self.dataset = self.dataset[self.dataset.size()[0] + data.size()[0]-self.DATASET_SIZE:]
            self.dataset_labels = self.dataset_labels[self.dataset_labels.size()[0] + data.size()[0]-self.DATASET_SIZE:]

        # Include data in dataset
        self.dataset = torch.cat((self.dataset, data))
        self.dataset_labels = torch.cat((self.dataset_labels, label))

        data_len = self.dataset.size()[1]
        lab_len = self.dataset_labels.size()[1]

        # concatenate data and label for shuffling
        new_data = torch.cat([self.dataset, self.dataset_labels], dim = 1)
        
        # train each model of the ensemble
        for model_i in range(self.ENSEMBLE_SIZE):
            # shuffle
            r = torch.randperm(data.size()[0])
            rand_data = new_data[r,:]

            #divide in batches
            data_batches = torch.utils.data.BatchSampler(rand_data, self.MODEL_BATCH_SIZE, False)

            losses = []

            for epoch in range(self.MODEL_EPOCHS): # 1 epoch = all data processed once

                for el in list(data_batches):

                    batch = torch.stack(el, 0)
                    # extract data and labels
                    data_batch = batch[:, :data_len]
                    label_batch = batch[:, data_len:data_len+lab_len-1]
                    rew_batch = batch[:, -1].unsqueeze(1)

                    # Update model using performed rollouts

                    criterion_model = torch.nn.MSELoss()
                    criterion_rew = torch.nn.MSELoss()

                    out_m = self.model[model_i](data_batch)
                    out_r = self.reward_net(data_batch)

                    self.adam_model[model_i].zero_grad()
                    self.adam_rew.zero_grad()

                    loss_model = criterion_model(out_m, label_batch)#.to(self.dev))
                    loss_rew = criterion_rew(out_r, rew_batch)#.to(self.dev))
                    
                    losses.append(loss_model.item())

                    #print("Model loss: {}".format(loss))

                    loss_model.backward()
                    loss_rew.backward()

                    self.adam_rew.step()
                    self.adam_model[model_i].step()
                    
            loss_dict["model_"+str(model_i)] = np.mean(losses)
            
        self.writer.add_scalars('Average models loss', loss_dict, t)
        
        return np.mean(losses)
        
    def model_step(self, obs, act, idx_model):
        
        # If the agent is model based, use the model
        if self.MODEL_BASED:
            observation = torch.tensor(obs).float().unsqueeze(dim=0)
            model_in = torch.cat((observation, act), 1)
            obs = None
            
            if idx_model == None:

                ens_obs = []

                for idx in range(self.ENSEMBLE_SIZE):
                    new_obs = self.model[idx](model_in)
                    new_obs = new_obs.cpu().detach().numpy()[0]
                    ens_obs.append(new_obs)

                # take avg over ensembles
                obs = np.mean(ens_obs, axis = 0)
            else:

                obs = self.model[idx_model](model_in)
                obs = obs.detach().cpu().numpy()[0]

            r = self.reward_net(model_in)
        
        action = None
        done = None
        
        if self.act_space == 'Discrete':
            action = int(act[0].cpu().item())
        else:
            action = act.cpu().item()
            if self.act_dim > 1:
                action = action.detach().numpy()
            else:
                action = [action]
        
        # If algorithm is model based, we just want the "done" function
        if self.MODEL_BASED:        
            #_,_,done,_ = self.env.step(action)
            # Only for Cartpole
            
            done = bool(
                obs[0] < -self.env.x_threshold
                or obs[0] > self.env.x_threshold
                or obs[2] < -self.env.theta_threshold_radians
                or obs[2] > self.env.theta_threshold_radians
            )
            
        else:
            obs, r, done, _ = self.env.step(action)
        
        return obs, r, done, action
        
    def rollout(self):
        
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_adv_obs = []
        batch_adv_acts = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []
        idx_model = None

        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

            # Reset the environment
            obs = self.env.reset()
            
            if self.MODEL_BASED:
                rnd = random.uniform(0,1)

                # With 0.5 chance, sample the starting state from the buffer
                if rnd >= 0.5:
                    idx = randrange(self.dataset.size()[0])
                    obs = self.dataset[idx, :self.dataset.size()[1]-1].cpu().numpy()
            
            prev_obs = None
            done = False

            # Run an episode for at most ROLLOUT_LEN timesteps
            for ep_t in range(self.ROLLOUT_LEN):

                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                batch_obs.append(obs)

                # Calculate action and make a step in the env. 
                action, log_prob = self.get_action(obs)
                prev_obs = obs
                
                # Perform adversarial choice of the model for this transition
                
                if self.ADVERSARIAL: 
                    idx_model, adv_obs = self.get_adversarial_action(obs, action)
                    batch_adv_obs.append(adv_obs)
                    batch_adv_acts.append(np.array([idx_model]))
                
                # Also returns the action converted to numpy 
                # TODO: edit to just use tensor
                
                obs, rew, done, action_np = self.model_step(obs, action, idx_model)
                
                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action_np)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if done or t>= self.timesteps_per_batch:
                    break
          
            # If rollout terminated because of short horizon, add state-value estimate to last rew to make up for it
            if not done:
                ep_rews[-1] += self.critic(prev_obs)
          
            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float, device = self.dev)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float, device = self.dev)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device = self.dev)
        batch_rtgs = self.compute_rtgs(batch_rews)
        batch_adv_acts = torch.tensor(batch_adv_acts, dtype=torch.float, device = self.dev)
        batch_adv_obs = torch.tensor(batch_adv_obs, dtype=torch.float, device = self.dev)
        
        #print("Average rollout length: {}".format(np.mean(batch_lens)))

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_adv_acts, batch_adv_obs
    
    def compute_rtgs(self, batch_rews):
        
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.GAMMA
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device = self.dev)

        return batch_rtgs
    
    def get_action(self, obs):
        
        mean = self.actor(obs)

        dist = None

        if self.act_space == 'Discrete':
            dist = Categorical(logits = mean)

            # Sample action from distribution
            action = dist.sample()

            # Compute log probability for action
            log_prob = dist.log_prob(action)

            # Return action and log probability
            return torch.tensor([action]).float().unsqueeze(dim=0), log_prob.detach()
        
        else:
        
            # Same as for Discrete, but with multidimensional/continuous action

            dist = MultivariateNormal(mean, self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            return torch.tensor(action).float().unsqueeze(dim=0), log_prob.detach()
        

    def get_adversarial_action(self, obs, action):
        
        obs = torch.tensor(obs).float().unsqueeze(dim=0)
        adv_in = torch.cat((obs, action), 1)
        
        epsilon = random.uniform(0,1)
        
        dist = None
        
        # Random action with epsilon probability
        if epsilon < self.EPSILON:
            
            mean = torch.tensor(np.ones(self.ENSEMBLE_SIZE))
            dist = Categorical(probs = mean)
        
        else:
            mean = self.adversary(adv_in)
            dist = Categorical(logits = mean)

        # Sample action from distribution
        action = dist.sample()

        # Compute log probability for action
        log_prob = dist.log_prob(action)

        # Return action and log probability
        return action, adv_in.detach().numpy()
        
    def evaluate(self, batch_obs, batch_acts):
        
        # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        
        mean = self.actor(batch_obs)
        dist = None

        if self.act_space == 'Discrete':
            dist = Categorical(logits = mean)

            # Compute log probability for action
            log_probs = dist.log_prob(batch_acts)#.to(self.dev))
        
        else:
        
            # Same as for Discrete, but with multidimensional/continuous action

            dist = MultivariateNormal(mean, self.cov_mat)
            log_probs = dist.log_prob(batch_acts)#.to(self.dev))

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs
        
    def learn(self, total_timesteps):
        """
          Train the actor and critic networks. Here is where the main PPO algorithm resides.
          Parameters:
            total_timesteps - the total number of timesteps to train for
          Return:
            None
        """
        #print(f"Learning... Running {self.ROLLOUT_LEN} timesteps per episode, ", end='')
        #print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        
        mod_loss = 0.0
        if self.MODEL_BASED:
            mod_loss = self.improve_model(i_so_far)
        
        while True:
          
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, adv_acts, adv_obs = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Calculate advantage at k-th iteration
            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                
                # Calculate V_phi and pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                
                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-torch.min(surr1, surr2)).mean()
                self.writer.add_scalar('Actor loss', actor_loss.item(), i_so_far)
                
                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                critic_loss = nn.MSELoss()(V, batch_rtgs)
                self.writer.add_scalar('Critic loss', critic_loss.item(), i_so_far)
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                
            # Calculate adversarial log probabilities of actions taken during rollouts
            # TODO: move this to separate function?
            if self.ADVERSARIAL:
                adv_batch_acts = self.adversary(adv_obs)
                
                # sample epsilon*100 percent of the transitions and make them random
                if self.EPSILON != 0:
                    indices = torch.tensor(random.sample(range(self.timesteps_per_batch), round(self.timesteps_per_batch*self.EPSILON)), device = self.dev)
                    adv_batch_acts[indices] = torch.ones(round(self.timesteps_per_batch*self.EPSILON),1,self.ENSEMBLE_SIZE, device = self.dev)

                dist = Categorical(logits = adv_batch_acts)
                adv_batch_log_probs = dist.log_prob(adv_acts).squeeze()

                # Adversarial REINFORCE loss: it is without "minus" because we want to minimize the discounted reward
                adv_loss = batch_rtgs * adv_batch_log_probs
                adv_loss = adv_loss.mean()
                self.writer.add_scalar('Adversary loss', adv_loss.item(), i_so_far)

                # Calculate gradients and perform backward propagation for adversarial network
                self.adv_optim.zero_grad()
                adv_loss.backward()
                self.adv_optim.step()
        
            if self.MODEL_BASED:
                mod_loss = self.improve_model(i_so_far)
                
            avg_rew = eval_policy(policy=self.actor, env=self.env, render=False, i = i_so_far, mod_loss = mod_loss)
            self.writer.add_scalar('Average reward', avg_rew, i_so_far)
            
            self.rewards_hist.append(avg_rew)

            if i_so_far == self.TOTAL_ITERATIONS:#400: #avg_rew >= 200:
                print("Terminating.")
                break