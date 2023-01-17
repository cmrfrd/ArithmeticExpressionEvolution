from evolve import Evolve, Agent, get_game
aa = Agent(4,get_game()).set_basic()
for i in range(100):
    aa.mutate()

def fuzz(agents=1, mutations=10000, stats_every=100, pause_every=500):
    for i in range(agents):
        aa = Agent(4,get_game())
        aa.set_basic()
        history = []
        for i in range(mutations):
            if i % stats_every == 0:
                history.append(aa.stats())
            aa.mutate()
            # logging.debug(aa())
            print(aa())
            #if (pause_every < mutations) and (i % pause_every == 0):
            #    for h in history:print(h)
            #    draw(aa)
            #    input("-- pause --")
fuzz()
import sys;sys.exit(0)

dict((k,list(v.children.values()) if hasattr(v, "children") else []) for k,v in aa.nodes.items())

NODES = [
    (Observable, 0.4),
    (Random, 0.0),
    (Constant, 0.1),
    (MulOp, 0.085),
    (AddOp, 0.085),
    (MinOp, 0.005),
    (MaxOp, 0.005),
    (DivOp, 0.08),
    (FDivOp, 0.08),
    (ModOp, 0.08),
    (ExpOp, 0.08)
]

MUTATE_GROW = [
    ## Add
    ("add_random_node", 0.27),
    ("add_random_edge", 0.14),
    ("add_random_unused_edge", 0.03),
    ## Delete
    ("delete_random_node_and_edges", 0.03),
    ("delete_random_edge", 0.03),
    ## Update
    ("update_random_edge", 0.1),
    ("update_weight", 0.4),
]

MUTATE_STAGGER = [
    ## Add
    ("add_random_node", 0.03),
    ("add_random_edge", 0.03),
    ("add_random_unused_edge", 0.03),
    ## Delete
    ("delete_random_node_and_edges", 0.03),
    ("delete_random_edge", 0.03),
    ## Update
    ("update_random_edge", 0.25),
    ("update_weight", 0.6),
]

MUTATE_SHRINK = [
    ## Add
    ("add_random_node", 0.16),
    ("add_random_edge", 0.03),
    ("add_random_unused_edge", 0.03),
    ## Delete
    ("delete_random_node_and_edges", 0.04),
    ("delete_random_edge", 0.04),
    ## Update
    ("update_random_edge", 0.1),
    ("update_weight", 0.6),
]

init_num_mutations = 100
num_mutations_per_generation = 50
population_size = 100
percent_survive = 0.1
MUTATE = MUTATE_GROW

E = Evolve(population_size=population_size, percent_survive=percent_survive)
E.reset_population(init_num_mutations=init_num_mutations)

generations = 100
for gen in range(generations):
    # if gen % 4 == 0: MUTATE = MUTATE_GROW
    # if gen % 4 == 1: MUTATE = MUTATE_STAGGER
    # if gen % 4 == 2: MUTATE = MUTATE_SHRINK
    # if gen % 4 == 3: MUTATE = MUTATE_GROW
    best_agent, best_agent_fitness = \
      E.run_generation(num_mutations=num_mutations_per_generation)
    print(f"Generation num: {gen}")
    print(f"Population best: {best_agent_fitness}")
    print(f"Best stats: \n{best_agent.stats()}")
    print()
    draw(best_agent)

for gen in range(100, generations+100):
    best_agent, best_agent_fitness = E.run_generation()
    print(f"Generation num: {gen}")
    print(f"Population best: {best_agent_fitness}")
    print(f"Stats: \n{best_agent.stats()}")
    print()
