import numpy as np
from typing import Generator, Any

import warnings
warnings.filterwarnings("ignore")

from utils import *
from nodes import *
from agent import *
from environments import get_env
from persist_dict import KV

from snakespace import SnakeSpace

def init_population(out_dim, population_size, init_num_mutations, nodes, mutations, env):
    for _ in range(population_size):
        yield Agent(nodes, mutations, env).set_basic(out_dim).mutate_n(init_num_mutations)

def mutate_agent(agent, num_mutations):
    return (
        agent.mutate_n(num_mutations)
    )

def get_agent_and_num_mutations(
      ordered_agents,
      agent_weighting,
      mutation_range,
      num_mutations_weighting,
    ) -> Generator[tuple[Any, int], None, None]:
    population_size = len(ordered_agents)

    sampled_agent_indexes = np.random.choice(
        a=range(population_size), size=population_size, p=agent_weighting
    )
    sampled_agent_mutations = np.random.choice(
        a=range(*mutation_range), size=population_size, p=num_mutations_weighting
    )

    for index in range(population_size):
        yield (
            ordered_agents[sampled_agent_indexes[index]],
            sampled_agent_mutations[index]
        )

def mutate_until_new_agent(old_agent, num_mutations, S_prefix, db_filename):
    ## mutate the old agent and make sure it hasn't been seen b4
    kv = KV(db_filename)
    new_agent = old_agent.copy().mutate_n(num_mutations)
    i = 1
    while S_prefix.s(new_agent.get_hash()) in kv:
        new_agent = new_agent.mutate_n(i)
        i += 1
    return new_agent

def p_mutate_new_agent(old_agent, num_mutations, env_str, S_prefix, db_filename):
    S_prefix = SnakeSpace(S_prefix, separator = '/')
    old_agent = deserialize(old_agent).set_env(get_env(env_str))
    new_agent = mutate_until_new_agent(old_agent, num_mutations, S_prefix, db_filename)
    return serialize(new_agent)
