from math import ceil, floor
from utils import rand_n_from_list, norm_vec
from agent import Agent

import numpy as np

def cutoff_selection_probability(agents_with_fitness: list[(Agent, float)], num_survivors) -> list[(Agent, float, float)]:
    agents_stats = list(agents_with_fitness)
    agents_fitness_sorted = sorted(agents_with_fitness, key=lambda i:i[1], reverse=True)

    selected_agents_probability = []
    for index, (agent, fitness) in enumerate(agents_fitness_sorted):
        if index < num_survivors:
            selected_agents_probability.append((agent, 1.0/num_survivors, fitness))
        else:
            selected_agents_probability.append((agent, 0, fitness))

    ## Sort final list by prob then fitness
    return sorted(selected_agents_probability, key=lambda i:(i[1],i[2]), reverse=True)


## return agents with probability of selection
def tournament_selection_probability(agents_with_fitness: list[(Agent, float)], num_survivors) -> list[(Agent, float, float)]:
    agents_stats = list(agents_with_fitness)

    partition_size = floor(len(agents_stats) / num_survivors)

    selected_agents_probability = []
    for p in range(num_survivors):

        ## Get the tournament winner, and the rest
        tournament,remaining = rand_n_from_list(partition_size, agents_stats)
        tournament_fitness_sorted = sorted(tournament, key=lambda i:i[1], reverse=True)
        tournament_winner, tornament_rest = tournament_fitness_sorted[0], tournament_fitness_sorted[1:]

        ## uniform chance of getting selected if a winner, (agent, prob, fitness)
        selected_agents_probability.append((tournament_winner[0], 1.0/num_survivors, tournament_winner[1]))
        selected_agents_probability += list(map(lambda e: (e[0], 0.0, e[1]),tornament_rest))

        ## Set the selection pool to what is remaining
        agents_stats = remaining

    ## Sort final list by prob then fitness
    return sorted(selected_agents_probability, key=lambda i:(i[1],i[2]), reverse=True)

def tournament_selection_with_diversity_probability(agents_with_fitness: list[(Agent, float)], num_survivors,
                                                    diversity_scale, fitness_scale) -> list[(Agent, float, float)]:
    agents_stats = list(agents_with_fitness)

    partition_size = ceil(len(agents_stats) / num_survivors)

    selected_agents_probability = []
    for p in range(num_survivors):

        ## Get the tournament
        tournament,remaining = rand_n_from_list(partition_size, agents_stats)

        ## Get the winner by the norm of <variation, fitness>
        vecs = (np.array(list(agent.stats().values())) for agent,_ in tournament)
        norm_vecs = list(norm_vec(v) for v in vecs)
        avg_vector = sum(norm_vecs)/len(norm_vecs)
        tournament_variation_fitness_metric = []
        for agent, fitness in tournament:
            agent_norm_vec = norm_vec(np.array(list(agent.stats().values())))
            agent_dist_from_avg_vec = np.linalg.norm(avg_vector - agent_norm_vec)
            tournament_variation_fitness_metric.append(
                (
                    agent,
                    (((diversity_scale*agent_dist_from_avg_vec)**2) + ((fitness_scale*fitness)**2)) ** 0.5
                )
            )
        tournament_variation_fitness_metric = sorted(tournament_variation_fitness_metric, key=lambda i:i[1], reverse=True)
        tournament_winner, tornament_rest = tournament_variation_fitness_metric[0], tournament_variation_fitness_metric[1:]

        ## uniform chance of getting selected if a winner, (agent, prob, metric)
        selected_agents_probability.append((tournament_winner[0], 1.0/num_survivors, tournament_winner[1]))
        selected_agents_probability += list(map(lambda e: (e[0], 0.0, e[1]),tornament_rest))

        ## Set the selection pool to what is remaining
        agents_stats = remaining

    ## Sort final list by prob then fitness
    return sorted(selected_agents_probability, key=lambda i:(i[1],i[2]), reverse=True)