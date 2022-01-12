# REAL-MBRL
Code for the MSc thesis project: REAL Reinforcement Learning - Planning with adversarial models.

## What is included

This repository contains two different implementations of the REAL agent:

- The first one features a tabular approach (Q-Learning), with a probabilistic model, and it is used to test the agent on the Frozen Lake environment. The code is available in the "Gridworld_code" folder.
- The second one features a Policy Gradient approach (PPO), with a deterministic model, and it is used to test the agent on the Cartpole and Pendulum environments. The code is available in the "Deterministic_model_code" folder.

To run the code, you just need to execute the "main.py" file (after installing the required libraries).

## How to edit (hyper)parameters

In the policy gradient approach, all the parameters/hyperparameters that you need to change are included in the Params.py file. Modify it to try new configurations.

In the tabular approach, the parameters/hyperparameters are located in the main.py file (since they are not too many).
