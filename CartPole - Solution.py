#Deep Cross Entropy
#https://colab.research.google.com/github/Cheukting/rl_workshop/blob/master/exercises/rl_workshop_deep_crossentropy_method.ipynb
'''
Instead of using a table (like for cross entropy use a function)
Start with random uniform probability for policy
loop:
    evaluate policy (using neural network - input:state out:action probabilities) n-times
    get top p% of evaluations and store the states and actions
    Use states and actions to adjust the neural network (so that it is more likely to take top_scorer actions from given states)
'''

import gym
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.loss import BCELoss, CrossEntropyLoss

class Network(nn.Module):
    def __init__(self, state_vars, actions):
        super(Network, self).__init__()
        self.layer1 = nn.Linear(in_features=state_vars, out_features=20)
        self.tanh = nn.Tanh()
        self.layer5 = nn.Linear(in_features=20, out_features=actions)
        


    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.tanh(self.layer5(x))
        x = f.softmax(x, dim=0)
        return x

def run_session(policy):
    states = []
    actions = []
    total_reward = 0

    done = False
    obs = env.reset()
    for _ in range(10000):
        probabilities = policy(torch.tensor(obs, dtype=torch.float)).detach().numpy() #use network to get action probabilities
        action = np.random.choice(np.arange(n_actions), p=probabilities) #Make a choice

        new_obs, reward, done, info = env.step(action)
        total_reward += reward

        states.append(obs)
        actions.append(action)

        obs = new_obs

        if done:
            break
    #print(total_reward)
    return states, actions, total_reward

def run_n_sessions(n_sessions, policy, percentile):

    #Track the generation
    n_state_list = []
    n_action_list = []
    n_total_rewards = []

    #Sample many times from the policy distribution and store the results
    for _ in range(n_sessions):
        obs_list, action_list, total_reward = run_session(policy)
        n_state_list.append(obs_list)
        n_action_list.append(action_list)
        n_total_rewards.append(total_reward)

    #Get best (state, action) pairs index
    reward_cutoff = np.percentile(n_total_rewards, percentile)
    sessions = np.argwhere(np.array(n_total_rewards) >= reward_cutoff).flatten()
    #print(top_scorers)

    top_states = []
    top_actions = []

    #for each top scorer store the states and the actions from those states that were taken
    for session in sessions:
        top_states += [state for state in n_state_list[session]]
        top_actions += [action for action in n_action_list[session]]

    #calculate average reward for generation
    gen_avg_reward = sum(n_total_rewards)/len(n_total_rewards)

    return top_states, top_actions, gen_avg_reward, reward_cutoff

def train(iterations, a_n, policy):
    
    x = []
    y = []

    n_sessions=100
    percentile=70
    learning_rate=0.001
    loss_fn = BCELoss()

    for i in range(iterations):
        s, a, r, cutoff = run_n_sessions(n_sessions, policy, percentile)


        #For plotting
        y.append(r)
        x.append(i)

        #Set up data for forward and back pass
        a = f.one_hot(torch.LongTensor(a), num_classes=a_n).float()
        s = torch.FloatTensor(s)

        #Use top_scorers (state, actions) to train the network
        for run in zip(s, a):
            pred = policy(run[0])
            target = run[1]
            loss = loss_fn(pred, target)
            policy.zero_grad()
            loss.backward()
            with torch.no_grad():
                for param in policy.parameters():
                    param -= learning_rate * param.grad

        print('Iteration: {} | Avg Reward: {} | Cutoff: {}'.format(i, round(r, 2), cutoff))
    
    return x, y, policy

def plot(x, y):
    plt.plot(x, y)
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.show()

def test_run(version):
    done = False
    obs = env.reset()
    total_reward = 0
    for _ in range(500):
        env.render()
        probabilities = policy(torch.tensor(obs, dtype=torch.float)).detach().numpy() #use network to get action probabilities
        action = np.random.choice(np.arange(n_actions), p=probabilities) #Make a choice

        new_obs, reward, done, info = env.step(action)
        total_reward += reward

        obs = new_obs

        if done:
            break
    env.close()
    print('{} Total Reward: {}'.format(version, total_reward))


#Initialize 
env_name = 'CartPole-v0'
env = gym.make(env_name).env
n_actions = env.action_space.n #number of actions
n_states = len(env.observation_space.sample()) #number of aspects for state
policy = Network(state_vars=n_states, actions=n_actions)

#Run
test_run('Pre-Training')
x, y, policy = train(100, n_actions, policy)
plot(x, y)
test_run('Post-Training')
