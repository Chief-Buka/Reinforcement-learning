import numpy as np
import math
import random
import matplotlib.pyplot as plt
import copy


#CONSTANTS
PHEADS = 0.4
THRESHOLD = 1e-10
DISCOUNT = 1

Reward = np.zeros((1, 101)).flatten()
Reward[100] = 1



def value_iteration():
    V = np.zeros( (1,101) ).flatten()
    all_V = []
    i = 1
    while True:
        delta = 0
        for s in range(1, V.shape[0]-1):
            old_value = V[s]
            best_value = 0
            for a in range(1, min(s, 100-s)+1):
                q = action_value(s, a, V)
                if q > best_value: 
                    best_value = q
            V[s] = best_value
            delta = max(delta, abs(old_value-V[s]))

        #For plotting
        if (i == 1 or i == 2 or i == 3):
            V_i = np.zeros( (1,101) ).flatten()
            np.copyto(V_i, V)
            all_V.append(V_i)

        #Convergence and Return
        if delta < THRESHOLD:
            all_V.append(V)
            return all_V, greedy_policy(V)
        
        i +=1



#act greedily with respect to policy
def greedy_policy(V):
    policy = dict()
    for s in range(1, V.shape[0]-1):
        q = [action_value(s, a, V) for a in range(1, min(s, 100-s)+1)]
        policy[s] = np.argmax(q)+1
    return policy


def plot_stuff(x, Y):
    for y in Y:
        plt.plot(x, y[:-1])
    plt.legend(['Iteration 1', 'Iteration 2', 'Iteration 3', 'Final Iteration'])
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.show()

def plot_stuff2(x, y):
    plt.bar(x, y)
    plt.xlabel('Capital')
    plt.ylabel('Value Estimates')
    plt.show()



def action_value(s, a, V):
    value = PHEADS*(Reward[s+a]+V[s+a]) + (1-PHEADS)*(Reward[s-a]+V[s-a]) #Value is weighted sum of resulting state
    return value


all_Value_Funcs, policy = value_iteration()
plot_stuff([s for s in range(100)], all_Value_Funcs)
plot_stuff2([s for s in range(99)], [policy[s] for s in policy])
