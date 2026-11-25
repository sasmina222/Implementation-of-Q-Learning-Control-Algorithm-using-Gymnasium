# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement



## Software Requirements



## Environment Description



## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm



## Python Program

```python

# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

for episode in range(episodes):

    state, info = env.reset()
    total_reward = 0

    for step in range(max_steps):

        action = choose_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        # Q-Learning Update
        if terminated or truncated:
            target = reward
        else:
            target = reward + gamma * np.max(Q[next_state])

        Q[state, action] += alpha * (
            target - Q[state, action]
        )

        state = next_state
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)

    # Epsilon Decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

rows = env.unwrapped.nrow
cols = env.unwrapped.ncol

def print_value_function(values):

    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(rows, cols), 3))


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [action_symbols[action] for action in policy]
    ).reshape(rows, cols)

    print("\nLearned Policy:")
    print(policy_grid)
# -------------------------------------------------
# Output
# -------------------------------------------------

state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)
print("Name: Sasmina S")
print("Reg no: 212225230254")
print("\nFinal Q-table:")
print(np.round(Q, 3))

print_value_function(state_values)
print_policy(learned_policy)
# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------
print("Name: Sasmina S")
print("Reg no: 212225230254")
window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Q-Learning Curve - FrozenLake")
plt.grid(True)
plt.show()

env.close()




```
---

## Output


Final Q-table:

Estimated State-Value Function:


<img width="298" height="510" alt="image" src="https://github.com/user-attachments/assets/d8527db7-6dc9-41d4-a861-5995b15f0f73" />




Learned Policy:

<img width="270" height="100" alt="image" src="https://github.com/user-attachments/assets/5ad3207d-0fe0-4854-a4eb-331b5f4385ba" />



Average reward over last 1000 episodes:

<img width="673" height="432" alt="image" src="https://github.com/user-attachments/assets/45cd0225-3d8e-4b59-ad64-49261b0b0ae1" />



---

## Result

```text



```

---

## Inference

```text



```

---

