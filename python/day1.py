from enum import Enum
import numpy as np
import operator
import random
from time import sleep
from collections import defaultdict


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
        # [::-1, :] transforms the given array by decart cordinate
        self.grid = grid[::-1, :]
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
        return self.grid[state.row][state.col] != Cell.BLOCK

    def cell(self, state):
        if not self.state_accessible(state):
            raise Exception(f'{state} is not accessible')
        return self.grid[state.row][state.col]

    @property
    def actions(self):
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        st = []
        for idxs, s in np.ndenumerate(self.grid):
            if s != Cell.B:
                st.append(State(*idxs))  # avoid containing block cells
        return st


class Agent():

    def __init__(self, env):
        self.actions = env.actions

    def policy(self, state):
        return random.choice(self.actions)


if __name__ == "__main__":
    P, G, B, F = Cell.PASS, Cell.GOAL, Cell.BLOCK, Cell.FALL

    grid = np.array([
        [P, P, P, G],
        [P, B, P, F],
        [P, P, P, P],
    ])

    env = Environment(grid)
    agent = Agent(env)

    n_episode = 10

    for i in range(n_episode):
        s = env.reset()
        total_reward = 0
        done = False

        while not done:
            a = agent.policy(s)
            next_s, r, done = env.step(a)
            total_reward += r
            s = next_s
            print(
                f'episode: {i}, state: {s}, action {a}, reward: {r}, total_reward: {total_reward}')
            # sleep(1)
