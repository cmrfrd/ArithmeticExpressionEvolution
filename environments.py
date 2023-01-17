import numpy as np
import gym
import gym_snake

def get_snake_env():
    return gym.make("Snake-8x8-v0")

def get_matrix_env(size=3, dim=2):
    X = np.random.uniform(0,8,size=[size]*dim)
    return X

def get_env(env_str):
    if env_str == "snake":
        return get_snake_env()
    elif env_str == "matrix":
        return get_matrix_env()
    elif env_str == "array":
        return get_matrix_env(dim=1)