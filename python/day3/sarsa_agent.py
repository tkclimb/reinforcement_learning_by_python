# %%
from collections import defaultdict
import gym
from python.day3.el_agent import ELAgent
from python.day3.utils import show_q_value


class SARSAAgent(ELAgent):
    def __init__(self, env, epsilon=0.1):
        super().__init__(env, epsilon)
        self.Q = defaultdict(lambda: [0] * len(self.actions))

    def learn(self, episode_count=100, gamma=0.9, learning_late=0.1, report_interval=50):
        self.init_log()
        for e in range(episode_count):
            s = self.env.reset()
            done = False
            a = self.policy(s)

            while not done:
                n_state, reward, done, _ = self.env.step(a)
                n_action = self.policy(n_state)  # On-policy
                # choose Q-vale given the next action selected by policy
                gain = reward + gamma * self.Q[n_state][n_action]
                # estimated value is from current Q-value
                estimated = self.Q[s][a]
                # update formula
                self.Q[s][a] += learning_late * (gain - estimated)
                s = n_state
                a = n_action
            else:
                self.log(reward)


def train():
    env = gym.make("FrozenLakeEasy-v0")
    agent = SARSAAgent(env, epsilon=0.1)
    agent.learn(episode_count=300)
    agent.show_reward_log()
    show_q_value(agent.Q)


if __name__ == "__main__":
    train()


# %%
