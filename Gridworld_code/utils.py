import numpy as np
def check_prediction(env):
    
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    for s in range(25-1):

        if s not in env.hlist:

            for action in range(4):

                print("For state {} and action {}".format(s,action))
                
                for i in range(env.ens_size):

                    print("Similarity for model {}: {}".format(i, np.array(env.sim_dict[i][s][action])))
            print("==================================================================")
            print("==================================================================")
    
def print_policy(agent, size):

    policy = []

    for i in range(size*size):

        action = np.argmax(agent.Q_table[i,:])
        if action == 0:
            policy.append("L")
        if action == 1:
            policy.append("D")
        if action == 2:
            policy.append("R")
        if action == 3:
            policy.append("U")

    policy = np.reshape(policy, (size,size))
    print(policy)