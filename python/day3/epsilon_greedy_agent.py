# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class CoinToss():

    def __init__(self, head_probs):
        self.head_probs = head_probs
        self.reset()

    def __len__(self):
        return len(self.head_probs)

    def reset(self):
        self.toss_count = 0

    def step(self, action):
        if action >= len(self.head_probs):
            raise Exception(f"The No.{action} coin doesn't exist")
        else:
            head_prob = self.head_probs[action]
            if random.random() < head_prob:
                reward = 1.0
            else:
                reward = 0.0
            self.toss_count += 1
            return reward


class EpsilonGreedyAgent():

    def __init__(self, env, epsilon, max_episode_steps):
        self.env = env
        self.epsilon = epsilon
        self.max_episode_steps = max_episode_steps

    def reset(self):
        self.env.reset()
        self.N = [0] * len(env)
        self.V = [0] * len(env)
        self.current_steps = 0

    def policy(self):
        coins = range(len(self.V))
        if random.random() < self.epsilon:
            return random.choice(coins)
        else:
            return np.argmax(self.V)

    def play(self):
        self.reset()
        rewards = []
        done = False

        for _ in range(self.max_episode_steps):
            selected_coin = self.policy()
            reward = env.step(selected_coin)
            rewards.append(reward)

            n = self.N[selected_coin]
            coin_average = self.V[selected_coin]
            new_average = (coin_average * n + reward) / (n + 1)
            self.N[selected_coin] += 1
            self.V[selected_coin] = new_average

        return rewards


if __name__ == "__main__":
    env = CoinToss([0.1, 0.5, 0.1, 0.9, 0.1])
    epsilons = [0.0, 0.1, 0.2, 0.5, ]
    game_steps = list(range(0, 110, 10))
    result = {}

    for eps in epsilons:
        means = []
        for steps in game_steps:
            agent = EpsilonGreedyAgent(env, eps, steps)
            rewards = agent.play()
            means.append(np.mean(rewards))
        result[f"epsilon = {eps}"] = means
    result["coin toss count"] = game_steps
    result = pd.DataFrame(result)
    result.set_index("coin toss count", drop=True, inplace=True)
    result.plot.line(figsize=(10, 5))
    plt.show()
