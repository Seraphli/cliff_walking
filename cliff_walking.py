import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from tabulate import tabulate

EPISODES = 500
RUNS = 10000

WORLD_HEIGHT = 4
WORLD_WIDTH = 12

START = (3, 0)
GOAL = (3, 11)

EPSILON = 0.1
ALPHA = 0.5
GAMMA = 1

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


class Env:
    def __init__(
        self, row=WORLD_HEIGHT, col=WORLD_WIDTH, start=START, goal=GOAL
    ) -> None:
        self._row = row
        self._col = col
        self._start = start
        self._goal = goal
        self._state = start
        self._total_reward = 0

    def step(self, action: int):
        y, x = self._state
        if action == ACTION_UP:
            state_ = [max(y - 1, 0), x]
        elif action == ACTION_DOWN:
            state_ = [min(y + 1, self._row - 1), x]
        elif action == ACTION_LEFT:
            state_ = [y, max(x - 1, 0)]
        elif action == ACTION_RIGHT:
            state_ = [y, min(x + 1, self._col - 1)]
        else:
            raise NotImplementedError()

        reward = -1
        if (action == ACTION_DOWN and y == 2 and 1 <= x <= 10) or (
            action == ACTION_RIGHT and y == self._start[0] and x == self._start[1]
        ):
            reward = -100
            state_ = self._start

        self._total_reward += reward
        self._state = state_

        return state_, reward

    @property
    def terminate(self):
        return self._state[0] == self._goal[0] and self._state[1] == self._goal[1]

    @property
    def state(self):
        return self._state

    @property
    def total_reward(self):
        return self._total_reward


class Learner:
    def __init__(
        self,
        row=WORLD_HEIGHT,
        col=WORLD_WIDTH,
        epsilon=EPSILON,
        alpha=ALPHA,
        gamma=GAMMA,
    ) -> None:
        self._epsilon = epsilon
        self._alpha = alpha
        self._gamma = gamma
        self._q = np.zeros((row, col, 4))

    def choose_action(self, state):
        if np.random.rand() < self._epsilon:
            return np.random.choice(ACTIONS)
        values_ = self._q[state[0], state[1], :]
        return np.random.choice(np.where(values_ == np.max(values_))[0])

    @property
    def Q(self):
        return self._q


class QLearning(Learner):
    def update(self, s, a, r, t, s_):
        self._q[s[0], s[1], a] += self._alpha * (
            r
            + (1 - int(t)) * self._gamma * np.max(self._q[s_[0], s_[1], :])
            - self._q[s[0], s[1], a]
        )


class SARSA(Learner):
    def update(self, s, a, r, t, s_, a_):
        self._q[s[0], s[1], a] += self._alpha * (
            r
            + (1 - int(t)) * self._gamma * self._q[s_[0], s_[1], a_]
            - self._q[s[0], s[1], a]
        )


def run_sarsa(learner: Learner):
    env = Env(WORLD_HEIGHT, WORLD_WIDTH, START, GOAL)
    state = env.state
    action = learner.choose_action(state)
    while not env.terminate:
        state_, reward = env.step(action)
        action_ = learner.choose_action(state_)
        learner.update(state, action, reward, env.terminate, state_, action_)
        state = state_
        action = action_
    return env.total_reward


def run_q_learning(learner: Learner):
    env = Env(WORLD_HEIGHT, WORLD_WIDTH, START, GOAL)
    state = env.state
    while not env.terminate:
        action = learner.choose_action(state)
        state_, reward = env.step(action)
        learner.update(state, action, reward, env.terminate, state_)
        state = state_
    return env.total_reward


def single_run(_):
    sarsa = SARSA()
    q_learning = QLearning()
    rewards_sarsa = np.zeros(EPISODES)
    rewards_q_learning = np.zeros(EPISODES)
    for i in range(EPISODES):
        rewards_sarsa[i] += run_sarsa(sarsa)
        rewards_q_learning[i] += run_q_learning(q_learning)
    return rewards_sarsa, rewards_q_learning, sarsa, q_learning


def print_optimal_policy(q_value):
    optimal_policy = []
    for y in range(WORLD_HEIGHT):
        optimal_policy.append([])
        for x in range(WORLD_WIDTH):
            if [y, x] == GOAL:
                optimal_policy[-1].append("G")
                continue
            bestAction = np.argmax(q_value[y, x, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append("U")
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append("D")
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append("L")
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append("R")
    print(tabulate(optimal_policy, tablefmt="heavy_grid"))


def main():
    rewards_sarsa = np.zeros(EPISODES)
    rewards_q_learning = np.zeros(EPISODES)
    with ProcessPoolExecutor() as executor:
        for r_sarsa, r_q_learning, sarsa, q_learning in tqdm(
            executor.map(single_run, range(RUNS)), total=RUNS
        ):
            for i in range(EPISODES):
                rewards_sarsa[i] += r_sarsa[i]
                rewards_q_learning[i] += r_q_learning[i]
    rewards_sarsa /= RUNS
    rewards_q_learning /= RUNS

    plt.plot(rewards_sarsa, label="Sarsa")
    plt.plot(rewards_q_learning, label="Q-Learning")
    plt.xlabel("Episodes")
    plt.ylabel("Sum of rewards during episode")
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig("images/figure_6_4.png")
    plt.close()

    print("Sarsa Optimal Policy:")
    print_optimal_policy(sarsa.Q)
    print("Q-Learning Optimal Policy:")
    print_optimal_policy(q_learning.Q)


if __name__ == "__main__":
    main()
