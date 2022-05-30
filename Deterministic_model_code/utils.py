import torch
from torch.distributions import Categorical, MultivariateNormal

def _log_summary(ep_len, ep_ret, ep_num, mod_loss):
    
    # Round decimal places for more aesthetic logging messages
    ep_len = format(ep_len, '.2f')
    ep_ret = format(ep_ret, '.2f')
    mod_loss = format(mod_loss, '.6f')
    
    
    ep_num = str(ep_num)
    ep_num = ep_num.zfill(3)
    
    # Print logging statements
    print(f"Episode #{ep_num:} | Avg Episode length: {ep_len:} | Avg Episode return: {ep_ret:} | Avg Model Loss: {mod_loss:} |")#, end="\r")

def rollout(policy, env, render):
    
  # Rollout until user kills process
  for n in range(100):
    obs = env.reset()
    done = False
    
    # number of timesteps so far
    t = 0

    # Logging data
    ep_len = 0            # episodic length
    ep_ret = 0            # episodic return
    
    while not done and t <=1000:
        t += 1
        action = policy(obs)
        
        if env.action_space.__class__.__name__ == 'Discrete':
            
            # Query deterministic action from policy and run it
            
            dist = Categorical(logits = action)

            # Sample an action from the distribution
            action = dist.sample()

            obs, rew, done, _ = env.step(action.cpu().item())

            # Sum all episodic rewards as we go along
            ep_ret += rew
        
        else:
            act_dim = env.action_space.shape[0]
            cov_var = torch.full(size=(act_dim,), fill_value=0.5)
            cov_mat = torch.diag(cov_var)
            
            dist = MultivariateNormal(action, cov_mat)
            
            action = dist.sample()
            
            if act_dim > 1:
                
                action = action.detach().numpy()
                
            else:
                
                action = [action.cpu().item()]
            
            obs, rew, done, _ = env.step(action)
            
            ep_ret += rew
        

    # Track episodic length
    ep_len = t
    # returns episodic length and return in this iteration
    yield ep_len, ep_ret

def eval_policy(policy= None, env = None, render=False, i = None, mod_loss = None):
  
    # Rollout with the policy and environment, and log each episode's data
    tot = 0
    avg_len = 0
    for ep_num, (ep_len, ep_ret) in enumerate(rollout(policy, env, render)):
        tot += ep_ret
        avg_len += ep_len
    _log_summary(ep_len=avg_len/100, ep_ret=tot/100, ep_num=i, mod_loss = mod_loss)
    return tot/100