# -*- coding:utf-8 -*-

"""

created by shuangquan.huang at 9/6/18

copied from open ai leader board

"""

import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

# states are: Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip
print('observation space:', env.observation_space)

# actions are: 0 left, 1 right
print('action space:', env.action_space)


class Policy():
    def __init__(self, s_size=4, a_size=2):
        self.w = 1e-4 * np.random.rand(s_size, a_size)  # weights for simple linear policy: state_space x action_space
    
    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x) / sum(np.exp(x))
    
    def act(self, state):
        probs = self.forward(state)
        # action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)  # option 2: deterministic policy
        return action


env.seed(0)
np.random.seed(0)

policy = Policy()


def hill_climbing(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-3):
    """Implementation of hill climbing with adaptive noise scaling.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    best_R = -np.Inf
    best_w = policy.w
    for i_episode in range(1, n_episodes + 1):
        rewards = []
        state = env.reset()
        
        for t in range(max_t):
            action = policy.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])
        
        if R >= best_R:  # found better weights
            best_R = R
            best_w = policy.w
            noise_scale = max(1e-3, noise_scale / 2)
            policy.w += noise_scale * np.random.rand(*policy.w.shape)
        else:  # did not find better weights
            noise_scale = min(2, noise_scale * 2)
            policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                       np.mean(scores_deque)))
            policy.w = best_w
            break
    
    return scores, i_episode - 100, np.mean(scores_deque)

# scores = []
# eps = float('inf')
# gm1, ns1 = 0, 0
#
# for gm in range(1, 11):
#     gm = gm / 10
#     for ns in [10, 100, 1000, 1000]:
#         ns = 1.0 / ns
#         policy = Policy()
#         s, e, _ = hill_climbing(noise_scale=ns, gamma=gm)
#         if e < eps:
#             eps = e
#             scores = s
#             print("params", gm, ns)
#             gm1 = gm
#             ns1 = ns


scores, _, _ = hill_climbing()


# print("="*80)
# print(gm1, ns1)
# print("="*80)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


state = env.reset()
for t in range(200):
    action = policy.act(state)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break

env.reset()
while True:
    
    env.render(mode="rgb_array")
env.close()
