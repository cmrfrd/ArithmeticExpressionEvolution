from cmath import inf
import sys
import tqdm

from nodes import *
from utils import *
from agent import *
from experiment import init_population
from fitness import fitness_sum
from selection import tournament_selection_probability

epsilon = 1e-10

NODES = [
    ("MatrixObservable", 0.5),
    ("AddOp", 0.5),
]

MUTATE = [
    ## Add
    ("add_random_node", 0.3),
    ("add_random_edge", 0.2),
    ## Delete
    ("delete_random_node_and_edges", 0.1),
    ("delete_random_edge", 0.2),
    ## Update
    ("update_random_edge", 0.1),
    ("update_weight", 0.1),
]

out_dim = 1 # don't change
init_num_mutations = 0
num_survivors = 64
survivor_mush_perc = 0.8
population_size = 512
num_generations = 32
mutation_range, mutation_mush_perc = (0,64), 0.75
agents = init_population(out_dim, population_size, init_num_mutations, NODES, MUTATE, env=None)

all_time = (None, -inf)
for gen in (gen_bar := tqdm.tqdm(range(num_generations), leave=None)):

    ## fitness
    agents_with_fitness: list[(Agent, float)] = [(agent, fitness_sum(agent)) for agent in agents]

    ## get agent probabilities and mush towards the apex
    agent_probabilities_fitness = tournament_selection_probability(agents_with_fitness, num_survivors)
    mushed_probabilities = mushed_weighting(get_i(agent_probabilities_fitness, 1), survivor_mush_perc)
    best_agent = agent_probabilities_fitness[0][0]
    best_agent_fitness = agent_probabilities_fitness[0][2]

    ## set all time if
    if (not all_time) or (best_agent_fitness > all_time[1]):
        all_time = (best_agent, best_agent_fitness)

    ## log
    gen_bar.set_description(f"Generation Progress: All time: {all_time[1]} Batch Best: {float(best_agent_fitness)}")

    # terminate condition
    if (-epsilon) < best_agent_fitness:
        print("Optimal found ...")
        sys.exit(0)

    ## sample the next agents, their number of mutations
    sampled_agent_indexes = np.random.choice(
        a=range(population_size), size=population_size, p=mushed_probabilities
    )
    sampled_agent_mutations = np.random.choice(
        a=range(*mutation_range), size=population_size, p=mushed_weighting_fixed(len(range(*mutation_range)), mutation_mush_perc)
    )

    ## create new agent pool with mutations
    old_agents = get_i(agent_probabilities_fitness, 0)
    new_agents = []
    for index in range(population_size):
        old_agent = old_agents[sampled_agent_indexes[index]]
        new_agent = old_agent.copy(None).mutate_n(sampled_agent_mutations[index])
        new_agents.append(new_agent)
    agents = new_agents

