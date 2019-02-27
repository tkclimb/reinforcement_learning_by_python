# %%
import numpy as np
from collections import defaultdict
import gym
from python.day3.el_agent import ELAgent
from python.day3.utils import show_q_value


class Actor(ELAgent):
    def __init__(self, env):
        super().__init__(env, epsilon=-1)  # negative epsilon suppresses random action
        row = env.observation_space.n
        col = env.action_space.n
        # initialize the Q-value by uniform distribution
        self.Q = np.random.uniform(0, 1, (row, col))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def policy(self, s):
        # stochastically choose the action based on the Q-value
        a = np.random.choice(self.actions, 1, p=self.softmax(self.Q[s]))
        return a[0]


class Critic():
    def __init__(self, env):
        n_states = env.observation_space.n
        self.V = np.zeros(n_states)


class ActorCritic():
    def __init__(self, env):
        self.env = env
        self.actor = Actor(env)
        self.critic = Critic(env)

    def train(self, episode_count=100, gamma=0.9, learning_late=0.1, report_interval=50):
        self.actor.init_log()

        for e in range(episode_count):
            s = self.env.reset()
            done = False
            while not done:
                a = self.actor.policy(s)
                next_state, reward, done, _ = self.env.step(a)
                gain = reward + gamma * self.critic.V[next_state]
                # estimated value is value of current state
                estimated = self.critic.V[s]
                td = gain - estimated
                self.actor.Q[s][a] += learning_late * td
                self.critic.V[s] += learning_late * td
                s = next_state
            else:
                self.actor.log(reward)

        return self.actor, self.critic


def train():
    env = gym.make("FrozenLakeEasy-v0")
    trainer = ActorCritic(env)
    actor, _ = trainer.train(episode_count=2500)
    actor.show_reward_log()
    show_q_value(actor.Q)


if __name__ == "__main__":
    train()


# %%
