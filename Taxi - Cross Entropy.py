# Cross Entropy
#https://colab.research.google.com/github/Cheukting/rl_workshop/blob/master/exercises/rl_workshop_crossentropy_method.ipynb
'''
Initialize a random probability distribution
loop:
    take n-samples from the distribution and evaluate them
    take the top p% (of results and corresponding values) 
    and use those to create the new distribution
The mean of the distribution should approach an optimal value
'''


import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env_name = 'Taxi-v3'
env = gym.make(env_name)

n_states = env.observation_space.n
n_actions = env.action_space.n



#Run one taxi epsiode
def run_session(policy):
    #initialize the episode
    done = False
    obs = env.reset()
    obs_list = []
    action_list = []
    total_reward = 0

    while not done:
        #env.render()
        action = random.choices(np.arange(n_actions), policy[obs], k=1)[0] # make a weighted random choice
        obs_list.append(obs)
        action_list.append(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        #print(reward)

    #print(total_reward)
    return obs_list, action_list, total_reward
    #env.close() 

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

    return top_states, top_actions, gen_avg_reward


def create_new_policy(top_states, top_actions):
    #use the top scorer (state, action) pairs to create a new distribution for the policy
    new_policy = np.zeros((n_states, n_actions))

    #accumulate the values (state, action) in new policy
    for i in range(len(top_states)):
        state = top_states[i]
        action = top_actions[i]
        new_policy[state][action] += 1

    #normalize the new policy
    for i in range(new_policy.shape[0]):
        total_actions = sum(new_policy[i])
        #if no actions were taken in that states make the probability random uniform
        if total_actions == 0:
            new_policy[i] = np.ones((1, n_actions))* 1./n_actions
        #otherwise normalize the counts for each state
        else:
            new_policy[i] = [count/total_actions for count in new_policy[i]]

    #print(new_policy)
    return new_policy

def run(iterations, n_sessions, percentile, learning_rate):  
    x = []
    y = []
    n_avg_rewards_list = []
    policy = np.ones((n_states, n_actions)) * 1./n_actions
    for i in range(iterations):
        top_states, top_actions, n_avg_rewards = run_n_sessions(n_sessions, policy=policy, percentile=percentile)
        n_avg_rewards_list.append(n_avg_rewards)
        new_policy = create_new_policy(top_states, top_actions)
        policy = new_policy * learning_rate + policy * (1-learning_rate) #take a weighted average between the old and new policy


        #For plotting the result
        x.append(i)
        y.append(n_avg_rewards)

        print('Iteration: {} | Average Reward: {} | Percentile: {} | Learning Rate: {}'.format(i, n_avg_rewards, percentile, learning_rate))

    return x, y


def plot_progession(x, y):
    plt.plot(x, y)
    plt.xlabel('Iteration #')
    plt.ylabel('Average Reward')
    plt.show()

# If percentile is too high, there is too much bias towards (state, action) pairs that don't do well because the distribution
# will be heavily biased towards only a few samples
# Essentially there wont be enough exploration 
x, y = run(iterations=100, n_sessions=250, percentile=30, learning_rate=.5)
plot_progession(x, y)
