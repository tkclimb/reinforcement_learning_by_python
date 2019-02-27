from enum import Enum
import numpy as np
import random
from abc import abstractmethod, ABCMeta
from time import sleep
from collections import defaultdict
from copy import copy
import pandas as pd
import matplotlib as plt


class State():

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __repr__(self):
        return f'<State: [{self.row}, {self.col}]>'

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    @property
    def idxs(self):
        return (self.row, self.col)


class Action(Enum):
    UP = 1
    DOWN = -1
    RIGHT = 2
    LEFT = -2


class Cell(Enum):
    PASS = 0
    GOAL = 1
    BLOCK = 9
    FALL = -1  # more correctly, pitfall


class Environment():

    def __init__(self, grid, move_prob=0.8):
        self.row = grid.shape[0]
        self.col = grid.shape[1]
        self.grid = grid
        self.default_reword = -0.04
        self.move_prob = move_prob
        self.reset()

    def reset(self):
        self.agent_state = State(0, 0)
        self.last_transition_prob = {}
        return self.agent_state

    def step(self, action):
        transition_probs = self.transit_func(self.agent_state, action)

        next_states = list(transition_probs.keys())
        probs = list(transition_probs.values())

        next_state = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(next_state)

        if not done:
            self.agent_state = next_state

        self.last_transition_prob = transition_probs
        return next_state, reward, done

    def transit_func(self, state, action):
        transition_probs = defaultdict(lambda: 0)

        if not self.can_action_at:
            return transition_probs

        opposite_direction = Action(action.value * -1)
        for a in self.actions:
            prob = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2

            next_state = self.move(state, a)
            transition_probs[next_state] += prob

        return transition_probs

    def reward_func(self, state):
        reward = self.default_reword
        done = False

        if self.cell(state) == Cell.GOAL:
            reward = 1
            done = True
        elif self.cell(state) == Cell.FALL:
            reward = -1
            done = True

        return reward, done

    def move(self, state, action):
        next_state = State(state.row, state.col)

        if action == Action.UP:
            next_state.row += 1
        elif action == Action.DOWN:
            next_state.row -= 1
        elif action == Action.LEFT:
            next_state.col -= -1
        else:  # Action.RIGHT
            next_state.col += 1

        # check whether the agent is not out of the grid
        if not self.state_accessible(next_state):
            next_state = state

        return next_state

    def can_action_at(self, state):
        return self.grid[state.row][state.col] == Cell.PASS

    def state_accessible(self, state):
        # This is necessary to check wheather the indices of the grid is not out of the range
        if not ((0 <= state.row < self.row) and (0 <= state.col < self.col)):
            return False
        # then, check the cell is not block
        return self.grid[state.idxs] != Cell.BLOCK

    def cell(self, state):
        if not self.state_accessible(state):
            raise Exception(f'{state} is not accessible')
        return self.grid[state.idxs]

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        st = []
        for idxs, s in np.ndenumerate(self.grid):
            if s != Cell.BLOCK:
                st.append(State(*idxs))  # avoid adding block cells
        return st


class Planner(metaclass=ABCMeta):
    def __init__(self, env):
        self.env = env
        self.logs = []

    def reset(self):
        self.env.reset()
        self.logs = []

    @abstractmethod
    def plan(self):
        pass

    def transition_at(self, state, action):
        transition_probs = self.env.transit_func(state, action)
        for next_state, prob in transition_probs.items():
            reward, _ = self.env.reward_func(next_state)
            yield next_state, prob, reward

    def record(self, V):
        self.logs.append(copy(V))


class ValueIterPlanner(Planner):
    def __init__(self, env, gamma=0.9, threshold=0.0001):
        super().__init__(env)
        self.gamma = gamma
        self.threshold = threshold

    def plan(self):
        self.reset()
        V = np.zeros_like(self.env.grid)

        while True:
            max_delta = 0
            self.record(V)

            for s in self.env.states:
                if not self.env.can_action_at(s):
                    continue

                expected_rewards = []
                for a in self.env.actions:
                    r = 0
                    for next_state, prob, reward in self.transition_at(s, a):
                        r += prob * (reward + self.gamma *
                                     V[next_state.idxs])
                    expected_rewards.append(r)
                max_reward = max(expected_rewards)
                delta = abs(max_reward - V[s.idxs])
                max_delta = max(max_delta, delta)
                # choose the max as the reward for the state
                V[s.idxs] = max_reward

            if delta < self.threshold:
                break
        return V


class PolicyIterPlanner(Planner):
    def __init__(self, env, gamma=0.9, threshold=0.0001):
        super().__init__(env)
        self.policy = defaultdict({})
        self.gamma = gamma
        self.threshold = threshold

    def reset(self):
        super().reset()
        policy_shape = (self.env.row, self.env.col, self.env.actions.size)
        self.policy = {}
        for s in self.env.states:
            self.policy[s] = {}
            for a in self.actions:
                self.policy[s][a] = 1 / self.env.actions.size

    def policy_evaluation(self):
        V = np.zeros_like(self.env.grid)
        while True:
            max_delta = 0
            for s in self.env.states:
                expected_rewards = []
                for a, a_prob in self.policy[s].items():
                    r = 0
                    for t_prob, next_state, reward in self.transition_at(s, a):
                        r += a_prob * t_prob * \
                            (reward + self.gamma * V[next_state])
                    expected_rewards.append(r)
                max_reward = max(expected_rewards)
                delta = abs(max_reward - V[s.idxs])
                max_delta = max(max_delta, delta)
                V[s.idxs] = max_reward

            if max_delta < self.threshold:
                break

    def plan(self):
        self.reset()

        def take_max_prob_action(value_action_dict):
            return max(value_action_dict, key=value_action_dict.get)

        while True:
            update_stable = True
            V = self.estimate_by_policy(self.gamma, self.threshold)
            self.record(V)

            for s in self.env.states:
                policy_action = take_max_prob_action(self.policy[s])
                action_rewards = {}

                for a in self.env.actions:
                    r = 0
                    for t_prob, next_state, reward in self.transition_at(s, a):
                        r += t_prob * (reward + self.gamma *
                                       V[next_state.idxs])
                    action_rewards[a] = r

                best_action = take_max_prob_action(action_rewards)
                if policy_action != best_action:
                    update_stable = False

                for a in self.policy[s]:  # update policy
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            if update_stable:
                break


class Agent():

    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        return random.choice(self.actions)


if __name__ == "__main__":
    P, G, B, F = Cell.PASS, Cell.GOAL, Cell.BLOCK, Cell.FALL

    grid = np.array([
        [P, P, P, P],
        [P, B, P, F],
        [P, P, P, G],
    ])

    env = Environment(grid)
    agent = Agent(env)

    n_episode = 10

    for i in range(n_episode):
        total_reward = 0
        done = False
        s = env.agent_state

        while not done:
            a = agent.policy(s)
            next_s, r, done = env.step(a)
            total_reward += r
            s = next_s
        print(f'episode: {i}, total_reward = {total_reward}')

    vp = ValueIterPlanner(env)
    v_grid = vp.plan()
    df = pd.DataFrame(v_grid)
    print('\nFinal Values are ')
    print(df)
