from cmath import inf
import sys;epsilon = 1e-10
import tqdm
from functools import partial
import multiprocessing as mp
from random import seed;seed(42)

from nodes import *
from utils import *
from agent import *
from experiment import (
    init_population, 
    get_agent_and_num_mutations,
    p_mutate_new_agent,
)
from fitness import fitness_determinent, fitness_sort
from selection import (
    tournament_selection_probability
)

NODES = [
    ("MatrixObservable", 0.3),
    ("MulOp", 0.3),
    ("IfLT", 0.3),
    ("Permute", 0.1)
]

MUTATE = [
    ## Add
    ("add_random_node", 0.1),
    ("add_random_edge", 0.1),
    ## Delete
    # ("delete_random_node_and_edges", 0.5),
    # ("delete_random_edge", 0.1),
    # ("prune", 0.1),
    ## Update
    ("update_random_edge", 0.1),
    ("update_weight", 0.2),
    ("update_random_node_type",0.2),
    ("update_head_edges", 0.3),
]

out_dim = 1 # don't change
init_num_mutations = 0
num_survivors = 16
survivor_mush_perc = 0.5
population_size = 512
num_generations = 512
mutation_range, mutation_mush_perc = (0,1), 0.5

agents = list(init_population(out_dim, population_size, init_num_mutations, NODES, MUTATE, env=get_matrix_env(dim=1)))
history = mp.Manager().dict()
lock = mp.Manager().Lock()
pool = mp.Pool(processes=16)

all_time = (None, -inf)
for gen in (gen_bar := tqdm.tqdm(range(num_generations), leave=None)):

    ## add agent to history and get the fitness
    agents_with_fitness: list[(Agent, float)] = []
    for agent in agents:
        fitness = fitness_sort(agent)
        agents_with_fitness.append((agent, fitness))

    ## replace a random agent with the all time
    agents_with_fitness[randint(0,len(agents_with_fitness)-1)] = all_time

    ## get agent survivor probabilities and mush distribution 
    ## towards the apex agent
    agent_probabilities_fitness = tournament_selection_probability(agents_with_fitness, num_survivors)
    mushed_probabilities = mushed_weighting(get_i(agent_probabilities_fitness, 1), survivor_mush_perc)
    best_agent = agent_probabilities_fitness[0][0]
    best_agent_fitness = agent_probabilities_fitness[0][2]

    ## set the all time
    if (not all_time) or (best_agent_fitness > all_time[1]):
        all_time = (best_agent, best_agent_fitness)

    ## Log
    gen_bar.set_description(f"Generation Progress: All time: {all_time[1]} Batch Best: {float(best_agent_fitness)}, History size: {len(history)})")

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
    # func = lambda l, item: (serialize(item[0]), item[1], "matrix", None, l)
    func = lambda l, item: (serialize(item[0]), item[1], "array", history, l)
    new_agents_results = pool.starmap(
        p_mutate_new_agent, 
        map(partial(func,lock), agent_and_num_mutations)
    )
    new_agents += list(map(deserialize,new_agents_results))

    ## current agent pool *are* the new agents
    agents = new_agents

