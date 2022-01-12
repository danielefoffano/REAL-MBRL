class Params:
    ENV_NAME = "CartPole-v1" # or 'Pendulum-v0'
    ALPHA = 5e-3        # learning rate
    MODEL_BATCH_SIZE = 200 # how many batches to divide the roll-out transitions into
    MODEL_EPOCHS = 25  # how many epochs we want to train model network
    #N_ROLLOUTS = 200
    
    GAMMA = 0.99        # discount rate
    POLICY_HIDDEN_SIZE = 64    # number of hidden nodes we have in policy
    ADV_POLICY_HIDDEN_SIZE = 32    # number of hidden nodes we have in adversary policy
    MODEL_HIDDEN_SIZE = 512     # number of hidden nodes we have in model
    REW_HIDDEN_SIZE = 256
    CRITIC_HIDDEN_SIZE = 128
    ROLLOUT_LEN = 15  #set to 300 for cartpole, 200 for pendulum
    PPO_BATCH_TIMESTEPS = 1000 # This is just for model free: 1000 for cartpole/pendulum
    ENSEMBLE_SIZE = 3 # number of models in the ensemble
    EPSILON = 0
    TIMESTEPS_PER_BATCH = 100000
    N_UPDATES_PER_ITERATION = 20
    LR = 0.0005
    CLIP = 0.2
    MODEL_BASED = True
    ADVERSARIAL = True

    DATASET_SIZE = 2000
    SAMPLES_FOR_MODEL = 1000
    MODEL_ROLLOUT_LEN = 1000
