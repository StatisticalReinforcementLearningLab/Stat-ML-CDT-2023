import numpy as np

def mean_reward(reward, transition):
    num_states, num_actions = transition.shape[0],  transition.shape[1]
    mean_reward = np.zeros((num_states, num_actions)) 
    for s in range(num_states):
        for a in range(num_actions):
            mean_reward[s, a] = np.sum(transition[s, a, :] * reward)
    return(mean_reward)

def q_iteration(mean_reward, transition, gamma, num_iterations):
    num_states, num_actions = mean_reward.shape

    # Initialize Q-function
    Q = np.zeros((num_states, num_actions))

    for _ in range(num_iterations):
        # Copy the current Q-function
        Q_prev = np.copy(Q)

        # Update Q-function for each state-action pair
        for s in range(num_states):
            for a in range(num_actions):
                # Compute the Q-value for the (s, a) pair
                q_value = mean_reward[s, a] + gamma * np.sum(transition[s, a, :] * np.max(Q_prev, axis=1))
                Q[s, a] = q_value

    # Compute optimal value function and policy
    V = np.max(Q, axis=1)
    policy = np.argmax(Q, axis=1)

    return Q, V, policy


# Example MDP with 3 states and 2 actions
num_states = 3
num_actions = 2

# Randomly initialize reward and transition matrices
reward = np.random.rand(num_states, num_actions)
transition = np.random.rand(num_states, num_actions, num_states)

NUM_ACTIONS = 2
NUM_STATES = 3
TRUE_REWARDS = np.array([0, 1, 10])
gamma = 0.95
delta = 0.1

TRANSITION_MATRIX = np.empty([NUM_STATES, NUM_ACTIONS,  NUM_STATES,])
TRANSITION_MATRIX[0, 0, :] = np.array([1-delta, 0, delta])
TRANSITION_MATRIX[1, 0, :] = np.array([1, 0, 0])
TRANSITION_MATRIX[2, 0, :] = np.array([1, 0, 0])
TRANSITION_MATRIX[0, 1, :] = np.array([0, 1, 0])
TRANSITION_MATRIX[1, 1, :] = np.array([0, 1, 0])
TRANSITION_MATRIX[2, 1, :] = np.array([0, 0, 1])

reward_matrix = mean_reward(TRUE_REWARDS, TRANSITION_MATRIX)
TRUE_REWARD_MATRIX = np.array([[10*delta, 1], [0, 1], [0, 10]])
print(TRUE_REWARD_MATRIX, reward_matrix)

# Set number of iterations
num_iterations = 100

# Run Q-iteration
Q, V, policy = q_iteration(reward_matrix, TRANSITION_MATRIX, gamma, num_iterations)

# Print the computed Q-function, value function, and policy
print("Q-function:")
print(Q)
print("Value function:")
print(V)
print("Policy:")
print(policy)

## OPTIMAL VSTAR COMPUTED BY HAND ##
VSTAR = np.zeros(NUM_STATES)
VSTAR[2] = TRUE_REWARDS[2]/(1-gamma) 
VSTAR[0] = delta * (TRUE_REWARDS[2] + gamma * VSTAR[2]) / (1-(1-delta)*gamma)
VSTAR[1] = gamma * VSTAR[0]
print(VSTAR)
