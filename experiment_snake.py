import sys;epsilon = 1e-10
from cmath import inf

import attrs
import tqdm
from functools import partial
import multiprocessing as mp
from random import seed;

from nodes import *
from utils import *
from agent import *
from persist_dict import KV
from experiment import (
    init_population, 
    get_agent_and_num_mutations,
    p_mutate_new_agent,
)
from fitness import fitness_snake
from environments import get_env
from selection import (
    tournament_selection_probability
)
from snakespace import SnakeSpace
S = SnakeSpace()
S.separator = '/'

NODES = [
    ("MatrixObservable", 0.2),
    ("ConstantInt", 0.1),
    ("AddOp", 0.2),
    ("SubOp", 0.2),
    ("MulOp", 0.2),
    ("DivOp", 0.2),
]

MUTATE = [
    ## Add
    ("add_random_node", 0.1),
    ("add_random_edge", 0.1),
    ## Delete
    ("delete_random_node_and_edges", 0.1),
    # ("delete_random_edge", 0.1),
    # ("prune", 0.1),
    ## Update
    ("update_random_edge", 0.1),
    ("update_weight", 0.1),
    ("update_head_edges", 0.5),
]

experiment_name = "approximate_snake"
run_id = make_name()
S_prefix = S.s(experiment_name).s(run_id)

db_filename = "kv.db"
kv = KV(db_filename)

out_dim = 1 # don't change
init_num_mutations = 0
num_survivors = 16
survivor_mush_perc = 0.5
population_size = 512
num_generations = 512
mutation_range, mutation_mush_perc = (0,1), 0.5

write_chunks = 16
agents = list(init_population(out_dim, population_size, init_num_mutations, NODES, MUTATE, env=get_env("snake")))
pool = mp.Pool(processes=4)

all_time = (None, -inf)
for gen in (gen_bar := tqdm.tqdm(range(num_generations), leave=None)):

    ## write agents in chunks
    for agent_chunk in chunks(agents, write_chunks):
        items = []
        for agent in agent_chunk:
            items.append((str(S_prefix.s(agent.get_hash())), serialize(agent)))
        kv.batch_set(items)

    ## add agent to history and get the fitness
    agents_with_fitness: list[(Agent, float)] = []
    for agent in agents:
        fitness = fitness_snake(agent)
        agents_with_fitness.append((agent, fitness))

    ## get agent survivor probabilities and mush distribution 
    ## towards the apex agent
    agent_probabilities_fitness = tournament_selection_probability(agents_with_fitness, num_survivors)
    mushed_probabilities = mushed_weighting(get_i(agent_probabilities_fitness, 1), survivor_mush_perc)
    best_agent = agent_probabilities_fitness[0][0]
    best_agent_fitness = float(agent_probabilities_fitness[0][2])

    ## set the all time
    if (not all_time) or (best_agent_fitness > all_time[1]):
        all_time = (best_agent, best_agent_fitness)

    ## Log
    gen_bar.set_description(f"Generation Progress: All time: {all_time[1]} Batch Best: {best_agent_fitness} Hist size: {kv.count_with_prefix(S_prefix)}")

    # terminate condition
    if (1-epsilon) < best_agent_fitness:
        print("Optimal found ...")
        sys.exit(0)

    ## sample the next agents and their number of mutations
    agent_and_num_mutations = get_agent_and_num_mutations(
        ordered_agents = get_i(agent_probabilities_fitness, 0),
        agent_weighting = mushed_probabilities,
        mutation_range = mutation_range,
        num_mutations_weighting = mushed_weighting_fixed(
            len(list(range(*mutation_range))), mutation_mush_perc
        ),
    )

    ## create new agent pool with mutations
    new_agents = []
    new_agents_results = pool.starmap(
        p_mutate_new_agent,
        map(lambda item: (serialize(item[0]), item[1], "snake", str(S_prefix), db_filename), agent_and_num_mutations)
    )
    new_agents += list(map(deserialize,new_agents_results))

    ## current agent pool *are* the new agents
    agents = new_agents
