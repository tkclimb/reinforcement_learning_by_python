# %%
from collections import defaultdict
import gym
from python.day3.el_agent import ELAgent
from python.day3.utils import show_q_value


class QLearnAgent(ELAgent):
    def __init__(self, env, epsilon=0.1):
        super().__init__(env, epsilon)
        self.Q = defaultdict(lambda: [0] * len(self.actions))

    def learn(self, episode_count, gamma=0.9,
              learning_rate=0.1, report_interval=50):
        self.init_log()

        for e in range(episode_count):
            s = env.reset()
            done = False

            while not done:
                a = self.policy(s)
                next_state, reward, done, _ = env.step(a)

                gain = reward + gamma * max(self.Q[next_state])
                estimated = self.Q[s][a]
                self.Q[s][a] += learning_rate * (gain - estimated)
                s = next_state
            else:
                self.log(reward)


def train():
    agent = QLearnAgent(env, epsilon=0.1)
    agent.learn(episode_count=300)
    # env.render()
    agent.show_reward_log()
    show_q_value(agent.Q)


if __name__ == "__main__":
    env = gym.make("FrozenLakeEasy-v0")
    train()


# %%
