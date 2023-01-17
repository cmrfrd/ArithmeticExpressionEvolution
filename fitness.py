from cmath import e, inf, pi
import numpy as np

from environments import get_matrix_env

def fitness_snake(agent, env):
    agent.set_env(env)
    try:
        sum_reward = 0
        sum_steps = 0
        observation = agent.env.reset()
        done = False
        c = 0
        while (not done) and c<1000:
            observation_next, reward, done, info = agent.env.step(
                np.argmax(agent()[0])
            )
            observation = observation_next
            sum_reward += reward
            sum_steps += 1
            c += 1
        # return (sum_reward) / ( len(agent.nodes) * c)
        return (sum_reward)
    except Exception as e:
        return -2 * num_games

def fitness_sum(agent, size: int = 5, repeats: int = 5):
    try:
        fitness_sum = 0.0
        for _ in range(repeats):
            X = np.random.random(size)
            y = X.sum()
            fitness_sum -= abs(agent.set_env(X)() - y)
        return float(fitness_sum / repeats)
    except:
        return -inf

def fitness_determinent(agent, size: int = 3, repeats: int = 16):
    try:
        fitness_sum = 0
        for _ in range(repeats):
            X = get_matrix_env(size)
            y = np.linalg.det(X)
            fitness_sum -= abs(agent.set_env(X)() - y)
        return e ** float(fitness_sum)
    except Exception as ex:
        return -1.0

def fitness_pi(agent):
    try:
        return e ** -abs(float(agent()[0]) - pi)
    except Exception as ex:
        return -1.0

def fitness_sort(agent, size: int = 10, repeats: int = 50):
    try:
        fitness_sum = 0
        for _ in range(repeats):
            X = get_matrix_env(size, dim = 1)
            y = sorted(X)
            fitness_sum -= abs(int(agent.set_env(X)() == y))
        return e ** float(fitness_sum)
    except Exception as ex:
        return -1.0
