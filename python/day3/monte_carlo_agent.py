# %%
import math
from collections import defaultdict
import gym
from python.day3.el_agent import ELAgent
from python.day3.utils import show_q_value


class MonteCarloAgent(ELAgent):

    def __init__(self, env, epsilon):
        super().__init__(env, epsilon)
        self.Q = defaultdict(lambda: [0] * len(self.actions))

    def learn(self, episode_count=100, gamma=0.9, report_interval=50):
        self.init_log()
        N = defaultdict(lambda: [0] * len(self.actions))

        for e in range(episode_count):
            s = self.env.reset()
            done = False
            experience = []

            while not done:
                a = self.policy(s)
                n_state, reward, done, _ = self.env.step(a)
                experience.append({"state": s, "action": a, "reward": reward})
                s = n_state
            else:
                self.log(reward)

            for i, x in enumerate(experience):
                s, a = x["state"], x["action"]

                G, t = 0, 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                N[s][a] += 1
                alpha = 1 / N[s][a]
                self.Q[s][a] = (1 - alpha) * self.Q[s][a] + alpha * G


def train():
    env = gym.make("FrozenLakeEasy-v0")
    agent = MonteCarloAgent(env, epsilon=0.1)
    agent.learn(episode_count=300)
    # env.render()
    agent.show_reward_log()
    show_q_value(agent.Q)


if __name__ == "__main__":
    train()
