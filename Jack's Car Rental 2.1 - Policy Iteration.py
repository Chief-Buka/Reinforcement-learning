import numpy as np
from scipy.stats import poisson


DISCOUNT = 0.9
CONVERGENCE_VALUE = 0.1
POISSON_MAX = 10
LOT1_REQ = 3
LOT1_RET = 3
LOT2_REQ = 4
LOT2_RET = 2

MOVING_REWARD = -2
RENTING_REWARD = 10
MAX_CARS = 20 + 1


def action_value(s1, s2, action, V):
    value = 0

    #reward for moving cars
    moving_reward = MOVING_REWARD*action

    #Generate probs before and just index into them
    probs_req1 = [poisson.pmf(num_request, mu=LOT1_REQ) for num_request in range(POISSON_MAX)]
    probs_req2 = [poisson.pmf(num_request, mu=LOT2_REQ) for num_request in range(POISSON_MAX)]
    probs_ret1 = [poisson.pmf(num_request, mu=LOT1_RET) for num_request in range(POISSON_MAX)] 
    probs_ret2 = [poisson.pmf(num_request, mu=LOT2_RET) for num_request in range(POISSON_MAX)]


    for lot1_req in range(min(POISSON_MAX, s1-action)):
         
        for lot2_req in range(min(POISSON_MAX, s2+action)):

            renting_reward = RENTING_REWARD*(lot1_req+lot2_req)
            
            for lot1_ret in range(POISSON_MAX ):
                
                for lot2_ret in range(POISSON_MAX ):

                    if s1-action-lot1_req+lot1_ret > 20:
                        new_s1 = 20
                    else:
                        new_s1 = s1-action-lot1_req+lot1_ret

                    if s2+action-lot2_req+lot2_ret > 20:
                        new_s2 = 20
                    else:
                        new_s2 = s2+action-lot2_req+lot2_ret
                    

                    #print(s1,s2, lot1_req, lot2_req, lot1_ret, lot2_ret, action)
                    probs = probs_req1[lot1_req] * probs_req2[lot2_req] * probs_ret1[lot1_ret] * probs_ret2[lot2_ret]
                    next_state_value = V[new_s1][new_s2]

                    value += probs*(moving_reward + renting_reward + DISCOUNT*next_state_value)

    return value



def policy_eval(V, policy):
    while True:
        delta = 0
        for s1 in range(len(V)):
            for s2 in range(len(V[s1])):
                old_value = V[s1][s2]
                V[s1][s2] = action_value(s1, s2, policy[s1][s2], V)
                delta = max(delta, abs(old_value-V[s1][s2]) )
        print(f'Max Difference: {delta}')
        if delta < CONVERGENCE_VALUE:
            print(f'Value function converged for policy')
            return V



def policy_improvement(V, policy):
    policy_converged = True
    for s1 in range(len(V)):
        for s2 in range(len(V[s1])):
            old_action = policy[s1][s2]
            stop = min(5, s1) + 1
            start = -min(5, s2)
            action_values = dict()
            for action in range(start, stop):
                action_values[action] = action_value(s1, s2, action, V)
            best_action = max(action_values, key=action_values.get) 
            policy[s1][s2] = best_action
            if best_action != old_action:
                #print(f'State ({s1}, {s2}), {old_action} -> {best_action}')
            
                policy_converged = False
    print(V)
    print(policy)
    return policy, policy_converged




def policy_iteration(V, policy):
    policy_converged = False
    i = 0
    while policy_converged == False:
        print(f'Iteration: {i}')
        V = policy_eval(V, policy)
        policy, policy_converged = policy_improvement(V, policy)
        i+=1
    return V, policy

Value_function = np.zeros((MAX_CARS, MAX_CARS))
Policy = np.zeros((MAX_CARS, MAX_CARS)).astype(int)

Value, Policy = policy_iteration(Value_function, Policy)

print(np.flip(Value, axis=0))
print(np.flip(Policy, axis=0))

