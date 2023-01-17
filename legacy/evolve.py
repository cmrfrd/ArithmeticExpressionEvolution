import tqdm
import cmath
import pickle
import bisect
import numpy as np
import string
from random import randint, choice, choices, uniform, shuffle, seed
from typing import Any, Callable
import attrs
from dataclasses import dataclass
import uuid
import operator as O
from math import e, floor
from copy import deepcopy
from functools import reduce
from itertools import chain
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import gym
from gym_snake.register import env_list
from multiprocessing import Pool

from utils import *
from nodes import *

import warnings
warnings.filterwarnings("ignore")

MAX_WORKERS=2

import logging
logger = logging.getLogger()
# logger.disabled = False

def fit(a):
    a = pickle.loads(a)
    return (a, agent_fitness(a))

def agent_mutate(arg):
    num_mutations, new_agent_num, new_mutant_agent = arg
    new_mutant_agent = pickle.loads(new_mutant_agent)
    for _ in range(num_mutations):
        new_mutant_agent.mutate()
    return new_mutant_agent

def agent_fitness(agent, num_games=1):
    try:
        sum_reward = 0
        sum_steps = 0
        for _ in range(num_games):
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
        return (sum_reward) / ( len(agent.nodes) * c)
    except Exception as e:
        print(f"Agent got exception: {e}")
        return -2 * num_games

def draw(agent):
    G = agent.to_networkx()
    plt.figure(figsize =(9, 9))
    colors = {
        "head": "red",
        "Constant": "green",
        "Observable": "green",
        }
    nx.draw(
        G,
        node_color=[colors.get(n.split('-')[0],"blue") for n in G],
        pos=graphviz_layout(G, prog="twopi"),
        node_size=1500,
        with_labels=True,
        arrows=True
        )
    plt.savefig("agent.png", format="PNG")

def get_game():
    return gym.make("Snake-4x4-v0")


def get_children(search_nodes, nodes, skip_head=False):
    seen = set()
    for search_node in search_nodes:
        search_node_name = search_node.name if hasattr(search_node, "name") else search_node
        if search_node_name in seen: continue
        search_iter = dfs(search_node_name, nodes)
        next(search_iter) if skip_head else None
        for search_iter_node_name in search_iter:
            if search_iter_node_name not in seen:
                yield search_iter_node_name
                seen.add(search_iter_node_name)

def dfs(node_name, nodes, include_path=False, additional_graph=None):
    if not additional_graph:
        additional_graph = RDict()
    def dfs_helper(search_node_name, path=None):
        if not path:
            path = []
        if search_node_name in path:raise Exception(f"Cycle: {path} {search_node_name}")
        search_node = nodes[search_node_name]
        path.append(search_node_name)
        yield path if include_path else path[-1]
        if hasattr(search_node, "children"):
            for _, (child_name, _) in search_node.children.items():
                yield from dfs_helper(child_name, path)
            for child_name in additional_graph.get(search_node.name, []):
                yield from dfs_helper(child_name, path)
        path.pop()
    yield from dfs_helper(node_name)

def copy_nodes(nodes):
    new_nodes = RDict()
    for node in nodes.values():
        new_nodes[node.name] = node.copy()
        if hasattr(node, 'children'):
            new_nodes[node.name].nodes = new_nodes
    return new_nodes

class Agent:
    def __init__(self, distance: int, env: Any):
        self.distance = distance
        self.nodes: dict[str, Node] = RDict()
        self.head: (str, Node) = ("head", None)
        self.env = env

    def copy(self, new_env):
        ## Create new agent and fill nodes
        new_agent = Agent(self.distance, new_env)
        new_agent.nodes = self.copy_nodes()
        new_agent.head = ('head', new_agent.nodes['head'])
        return new_agent

    def to_networkx(self):
        G = nx.DiGraph()
        for node_name, node in self.nodes.items():
            G.add_node(node_name)
            if hasattr(node, 'children'):
                for _,(c_name,_) in self.nodes[node_name].children.items():
                    G.add_node(c_name)
                    G.add_edge(node_name, c_name)
        return G

    def set_basic(self):
        nodes = [
            LinearOp(
                name="head",
                ins_range=(0,3),
                outs=3,
                children=RDict(**{
                    0:("add1",0),
                    1:("add2",0),
                    2:("add3",0),
                })
            ).reset(),
            AddOp(
                name="add1",
                children=RDict(**{
                    0:("c1",0),
                })
            ),
            AddOp(
                name="add2",
                children=RDict(**{
                    0:("c2",0),
                })
            ),
            AddOp(
                name="add3",
                children=RDict(**{
                    0:("c3",0),
                })
            ),
            Constant(name="c1", weights=[0.1]),
            Constant(name="c2", weights=[0.1]),
            Constant(name="c3", weights=[0.1]),
        ]
        self.add_nodes(nodes)
        return self

    def stats(self):
        total_nodes = 0
        total_constants = 0
        total_nonconstants = 0
        total_edges = 0
        total_unused_inputs = 0
        total_capacity = 0
        unique_locations = set()
        total_num_locations = ((2*self.distance+1)**2) * 3
        for node_name, n in self.nodes.items():
            total_nodes += 1
            if hasattr(n, "children"):
                total_edges += len(n.children)
                total_unused_inputs += n.ins_range[1] - n.ins_range[0] - len(n.children)
                total_nonconstants += 1
            else:
                total_edges += 0
                total_constants += 1

                if n.__class__.__name__ == "Observable":
                    unique_locations.add((n.x,n.y,n.z))

            total_capacity += n.ins_range[1] - n.ins_range[0]
            if n.__class__ == Constant: total_constants += 1

        return dict(
            total_nodes=total_nodes,
            total_constants=total_constants,
            total_nonconstants=total_nonconstants,
            total_capacity=total_capacity,
            current_capacity=round(total_edges / total_capacity, 3),
            total_edges=total_edges,
            total_unused_inputs=total_unused_inputs,
            perc_env_observed=round(len(unique_locations)/total_num_locations, 3),
            deepest=max(map(len, dfs("head", self.nodes, include_path=True))),
            num_head_children=len(set(n for n in dfs("head", self.nodes))),
        )

    def mutate(self):
        mutation_op = np.random.choice([n for n,p in MUTATE], p=[p for n,p in MUTATE])
        print(f"Performing {mutation_op} mutation")
        getattr(self, mutation_op)()

    def add_nodes(self, nodes):
        assert "head" in map(lambda n:n.name, nodes), "One node must be the head"

        for n in nodes:
            if hasattr(n, "children"):
                n.nodes = self.nodes
            if n.name == "head":
                self.head = ("head", n)
            self.nodes[n.name] = n

    def num_nodes(self):
        return len(self.nodes)

    def node_discrepency(self):
        for node in dfs("head", self.nodes):
            if node.name not in self.nodes:
                raise Exception(f"Searched node {node.name} not in nodes")

    def unrefed_nodes(self):
        visited = set()
        for node in dfs("head", self.nodes):
            visited.add(node.name)

        if len(visited) > len(self.nodes):
            raise Exception(f"Node Discrepency: {len(self.nodes)} known, {len(visited)} found")

    def get_node(self, k=None, avoid=None):
        if not k:
            select_nodes = copy_nodes(self.nodes)
            select_nodes.pop(self.head[0])

            if not avoid:
                ky = select_nodes.random_key()
                return ky, self.nodes[ky]
            else:
                a = select_nodes.random_key()
                b = self.nodes[select_nodes.random_key()]
                while b == avoid:
                    a = select_nodes.random_key()
                    b = self.nodes[select_nodes.random_key()]
                return a,b
        return k, self.nodes[k]

    def get_parents_of_node(self, node):
        parents_unique_names = set()
        for search_node_name, search_node in self.nodes.items():
            for path in dfs(search_node_name, self.nodes, include_path=True):
                ## found path to the target node
                if path[-1] == node:
                    ## yield all the new paths nodes names and add them to the set
                    path_unique_names = set(path)
                    new_parents_unique_names = path_unique_names - parents_unique_names
                    yield from (np_name for np_name in new_parents_unique_names)
                    parents_unique_names.update(new_parents_unique_names)

    def get_refs_of_node(self, node):
        '''Yield nodes whose child is "node"'''
        for search_node_name, search_node in self.nodes.items():
            if hasattr(search_node, "children"):
                for child_num, (child_name, out) in search_node.children.items():
                    if node.name == child_name:
                        yield (search_node, child_num)

    def get_random_output(self, nodes, avoid_children=None, avoid_static=None):
        if not avoid_children:avoid_children=[]
        if not avoid_static:avoid_static=[]
        select_nodes = copy_nodes(nodes)
        select_nodes.pop("head")
        select_nodes.remove_keys(
            list(
             node_name
             for node_name in chain(get_children(avoid_children, select_nodes), avoid_static)
            )
        )
        total_num_outs = sum(node.outs for name, node in select_nodes.items())
        if total_num_outs == 0: return (None, None)
        rand_out = randint(1, total_num_outs)
        out_node = None
        for _, n in select_nodes.items():
            if rand_out - (n.outs) <= 0:
                out_node = n
                break
            rand_out -= n.outs
        return nodes[out_node.name], rand_out-1

    def get_random_input(self, nodes, avoid_children=None, avoid_static=None):
        if not avoid_children:avoid_children=[]
        if not avoid_static:avoid_static=[]
        select_nodes = copy_nodes(nodes)
        select_nodes.pop("head")
        select_nodes.remove_keys(
            list(
             node_name
             for node_name in chain(get_children(avoid_children, select_nodes), avoid_static)
            )
        )
        total_num_ins = sum(node.ins_range[1] - node.ins_range[0] for name, node in select_nodes.items())
        if total_num_ins == 0: return (None, None)
        rand_in = randint(1, total_num_ins)
        in_node = None
        for _, n in select_nodes.items():
            if rand_in - (n.ins_range[1] - n.ins_range[0]) <= 0:
                in_node = n
                break
            rand_in -= (n.ins_range[1] - n.ins_range[0])
        return nodes[in_node.name], rand_in-1

    def get_random_unused_input(self, nodes, avoid_children=None, avoid_static=None):
        if not avoid_children:avoid_children=[]
        if not avoid_static:avoid_static=[]
        select_nodes = copy_nodes(nodes)
        print(select_nodes.keys())
        select_nodes.pop("head")
        select_nodes.remove_keys(
            list(
             node_name
             for node_name in chain(get_children(avoid_children, select_nodes), avoid_static)
            )
        )
        total_num_ins = 0
        for name, node in select_nodes.items():
            if hasattr(node, "children"):
                unused_inputs = node.ins_range[1] - node.ins_range[0] - len(node.children)
                if unused_inputs < 0:
                    raise Exception(f"Negative unused inputs for node: {node}")
                total_num_ins += unused_inputs

        if total_num_ins == 0: return (None, None)
        print(f"Total num ins unused: {total_num_ins}")
        rand_in = randint(1, total_num_ins)
        in_node = None
        for _, n in select_nodes.items():
            if hasattr(n, "children"):
                if rand_in - (n.ins_range[1] - n.ins_range[0] - len(n.children)) <= 0:
                    in_node = n
                    break
                rand_in -= (n.ins_range[1] - n.ins_range[0] - len(n.children))

        ## Get the "rand_in"th input of "in_node"
        available_node_ins = RDict(
            **dict(
                (i, True) if i not in in_node.children else (-1, None)
                for i in range(in_node.ins_range[1] - in_node.ins_range[0])
            )
        )
        if -1 in available_node_ins:del available_node_ins[-1]

        key = nth(available_node_ins.keys(), rand_in-1)

        return nodes[in_node.name], key

    def add_random_node(self):
        """Make a node, and connect random inputs"""

        ## Make node
        node_class = np.random.choice([n for n,p in NODES], p=[p for n,p in NODES])
        if hasattr(node_class,"children"):
            new_node = node_class(
                name=node_class.__name__+'-'+make_name(),
                nodes=self.nodes,
                children=RDict()
            )
        elif node_class.__name__ == Observable.__name__:
            new_node = node_class(
                name=node_class.__name__+'-'+make_name(),
                max_distance=self.distance,
                env = self.env,
                x=0,y=0,z=0,
            ).mutate()
        else:
            new_node = node_class(
                name=node_class.__name__+'-'+make_name()
            )

        if new_node.name in self.nodes:
            raise Exception("Node name already exists!")
        new_node.mutate()

        ## If there is atleast one input
        new_node_in_diff = new_node.ins_range[1] - new_node.ins_range[0]
        if (new_node_in_diff) > 0:

            ## Wire the outputs of some nodes in the input of this one
            outputs = []
            num_outputs_to_wire = randint(new_node.ins_atleast, new_node_in_diff)
            print(f"Sampling {num_outputs_to_wire} outputs to input into new node {new_node.__class__}")
            # print(f"New node: \n {new_node}")
            if len(new_node.children):
                raise Exception("New node has children!?")

            for out in range(num_outputs_to_wire):
                # print(f"Finding out num {out}")

                ## Use then to find an output to "wire in"
                out_node, num_out = self.get_random_output(
                    self.nodes,
                    avoid_static=[new_node.name]
                )

                if (not out_node):
                    print("No available out node")
                    return (False, False)
                if (out_node == new_node):
                    print("Out node IS the new node")
                    return (False, False)

                outputs.append(out_node)
                new_node.children[out] = (out_node.name, num_out)

            # if len(outputs) != num_outputs_to_wire: raise Exception(f"Ouputs don't match, {len(outputs)}, {num_outputs_to_wire}")

            # for i, (n,o) in enumerate(outputs):
            #     node.children[i] = (n,o)

            print("Outs -> Node[in]")
            print(f"Connected {num_outputs_to_wire} outputs of {[(o.name,o.__class__) for o in outputs]} \ninto node {new_node.name}")
            if len(new_node.children) != num_outputs_to_wire:
                raise Exception(f"Ouputs don't match {len(new_node.children)}, {num_outputs_to_wire}")

        else:
            print(f"Generated node has no inputs")

        ## Take the output(s) of this node and feed it into some other node
        nono_inputs = [new_node.name]
        num_outputs_to_wire = randint(1, new_node.outs)
        for inp in range(num_outputs_to_wire):
            print(f"Finding inp num {inp}")

            ## ensure no cycles!
            new_nodes = RDict(**self.nodes, **{new_node.name:new_node})
            in_node, num_in = self.get_random_unused_input(
                nodes=new_nodes,
                avoid_children=nono_inputs,
                avoid_static=get_children(
                    [new_node],
                    new_nodes
                )
            )

            if not in_node:
                print(f"adding dead terminal node: {new_node.name}")
                self.nodes[new_node.name] = new_node
                return (True, False)
            if in_node == new_node:
                print(f"Random unused input IS node: {new_node.name}")
                self.nodes[new_node.name] = new_node
                return (True, False)

            in_node.children[num_in] = (new_node.name, inp)
            nono_inputs.append(in_node)
            print("Node[out] -> Ins")
            print(f"Connected output of new node {new_node.name} \n to inputs {(in_node.name,in_node.__class__,num_in)}")
        self.nodes[new_node.name] = new_node
        return (True, True)

    def add_random_unused_edge(self):
        out_node, num_out = self.get_random_output(self.nodes)
        in_node, num_in = self.get_random_unused_input(self.nodes, avoid_children=[out_node])
        if not in_node:
            print(f"No free inputs")
            return False
        in_node.children[num_in] = (out_node.name, num_out)
        return True

    def add_random_edge(self):
        out_node, num_out = self.get_random_output(self.nodes)
        in_node, num_in = self.get_random_input(self.nodes, avoid_children=[out_node])
        if not in_node:
            print(f"No free inputs")
            return False
        print(f"Adding edge/child into node: {in_node.name, [(c_name) for _,(c_name,_) in in_node.children.items()]}")
        print(f"slot num {num_in}")
        print(f"From output {out_node.name, num_out}")
        in_node.children[num_in] = (out_node.name, num_out)
        return True

    def update_random_edge(self):
        in_node, num_in = self.get_random_input(self.nodes)
        if not in_node:
            print(f"No free inputs")
            return False

        influenced_nodes_names = uniques(
            p_name for p_name in self.get_parents_of_node(in_node)
        )

        ## Try to find a random output the "search node" can "hook" into
        out_node, num_out = self.get_random_output(self.nodes, avoid_static=influenced_nodes_names)
        if not out_node:
            print(f"Sampled node {in_node.name} ")
            return False
        in_node.children[num_in] = (out_node.name, num_out)
        return True

    def update_weight(self):
        weight_nodes_names = list(filter(lambda k: hasattr(self.nodes[k],"weights"),self.nodes.keys()))
        weight_node_name = choice(weight_nodes_names)
        self.nodes[weight_node_name].mutate()
        return True

    def delete_random_edge(self):
        nodes_names_with_spare_edges = []
        for node_name, node in self.nodes.items():
            if (node.ins_range[1] - node.ins_range[0]) == 0:
                continue
            if len(node.children) > node.ins_atleast:
                nodes_names_with_spare_edges += [(node_name, child_id) for child_id in node.children.keys()]

        if len(nodes_names_with_spare_edges) == 0:
            print("No nodes with spare edges")
            return False

        n_name, child_id_to_delete = choice(nodes_names_with_spare_edges)
        del self.nodes[n_name].children[child_id_to_delete]
        return True

    def delete_random_node_and_edges(self, node_name=None):
        if not node_name:
            tobe_deleted_name, tobe_deleted_node = self.get_node()
        else:
            tobe_deleted_name, tobe_deleted_node = node_name, self.nodes[node_name]

        ## ensure that there is atleast one terminal node
        if (tobe_deleted_node.ins_atleast==0) and iter_size(filter(lambda n:n.ins_atleast==0, self.nodes.values()))==1:
            print(f"Dag needs atleast one terminal node")
            return False

        print(f"deleting node {tobe_deleted_node.name} with {len(tobe_deleted_node.children) if hasattr(tobe_deleted_node, 'children') else 0}  children")
        delete_actions = []
        reassign_actions = [] # (search_node, inp_num, (new_child, out))
        for ref_node, ref_node_child_num in self.get_refs_of_node(tobe_deleted_node):
            ## If there are enough children in the search node
            ## to operate, then it's safe to delete the edge
            ref_node_deletes = iter_size(filter(lambda item: item[0]==ref_node,delete_actions))
            if (len(ref_node.children)-ref_node_deletes) > ref_node.ins_atleast:
                # del node.children[inp_num]
                delete_actions.append((ref_node, ref_node_child_num))
                continue

            ## If the to be deleted node is *not* terminal
            ## reroute the inputs from the to be deleted node, into this searched one
            if tobe_deleted_node.ins_atleast > 0:
                new_child_name, (new_child, out) = tobe_deleted_node.children.random_item()
                reassign_actions.append((ref_node, ref_node_child_num, (new_child, out)))
                continue

            ## 1. The to be deleted node *is* terminal
            ## 2. search node edge can't be deleted, it *must* be rerouted to an output
            ## <- get the nodes who need "search node" to be evaluated
            ##    including those that will potentially be reassigned

            ## Get
            reassign_dict = RDict()
            for rn, rn_child_num, (nc, o) in reassign_actions:
                if rn.name not in reassign_dict: reassign_dict[rn.name] = [nc.name]
                else: reassign_dict[rn.name].append(nc.name)
            recently_reassigned_nodes_whose_children_contain_current_ref = uniques(
                map(
                    lambda item: item[0].name,
                    filter(
                        lambda item: ( ref_node.name in dfs(item[2][0], self.nodes, additional_graph=reassign_dict) ),
                        reassign_actions
                    )
                )
            )

            nodes_recently_reassigned_to_current_ref = \
              map(lambda item: item[0],filter(lambda item: item[2][0].name == ref_node.name,reassign_actions))
            parents_of_nodes_recently_reassigned_to_current_ref = uniques(
                p_name
                for n in nodes_recently_reassigned_to_current_ref
                for p_name in self.get_parents_of_node(n)
            )

            influenced_nodes_names = uniques(
                p_name for p_name in self.get_parents_of_node(ref_node)
            )

            influenced_nodes = chain(
                recently_reassigned_nodes_whose_children_contain_current_ref,
                influenced_nodes_names,
                parents_of_nodes_recently_reassigned_to_current_ref,
                [tobe_deleted_node.name]
            )

            ## Try to find a random output the "search node" can "hook" into
            out_node, num_out = self.get_random_output(
                self.nodes,
                avoid_static=influenced_nodes
            )
            if not out_node:
                print(f"Sampled node {tobe_deleted_name} is to important to be deleted")
                return False

            ## Reassign the ref node edge
            reassign_actions.append((ref_node, ref_node_child_num, (out_node, num_out)))
            continue

        if len(delete_actions) == 0 and len(reassign_actions) == 0:
            del self.nodes[tobe_deleted_name]
            print(f"Node {tobe_deleted_name} is terminal")
            return True

        ## Run all the actions
        for n, d_inp_num in delete_actions:
            del n.children[d_inp_num]
            print(f"Deleting ({n.name}, child -> {d_inp_num})")
        for n, r_inp_num, (out_node, num_out) in reassign_actions:
            n.children[r_inp_num] = (out_node, num_out)
            print(f"Reassigning input ({n}, {r_inp_num}) to output ({out_node},{num_out})")
        print(f"{len(delete_actions)} edge deletes, {len(reassign_actions)} edge reassings")

        del self.nodes[tobe_deleted_name]
        return True

    def find_cycle(self):
        visiting = {}
        result = []
        def topoDFS(node):
            visiting[node.name] = 1
            if hasattr(node, "children"):
                for _, (child, _) in node.children.items():
                    if child.name in visiting:
                        if visiting[child.name] == 2:
                            continue
                        if visiting[child.name] == 1:
                            raise Exception("Cycle")
                    topoDFS(child)
            visiting[node.name] = 2
            result.append(node.name)
        topoDFS(self.head[1])
        return result

    def __call__(self, cache=True):
        if cache:
            result = self.head[1]()
            clear_cache()
        else:
            CACHE_ENABLED = False
            result = self.head[1]()
            clear_cache()
        return result

@attrs.define
class Evolve:

    population_size: int
    percent_survive: float

    population: dict[int, Agent] = {}

    def basic_agent(self):
        a = Agent(4,get_game())
        a.set_basic()
        return a

    def reset_population(self, init_num_mutations, *args, **kwargs):

        def make_basic_agents():
            for agent_num in range(self.population_size):
                yield (
                    init_num_mutations,
                    agent_num,
                    pickle.dumps(self.basic_agent()),
                )

        new_population = {}
        with Pool(processes=MAX_WORKERS) as pool:
            results = tqdm.tqdm(
                enumerate(pool.imap_unordered(agent_mutate, list(make_basic_agents()))),
                total=self.population_size
            )
            for (i,a) in results:
                new_population[i] = a
        self.population = new_population

    def run_generation(self, num_mutations: int):
        with Pool(processes=MAX_WORKERS) as pool:
            print(f"Getting best agents by fitness ...")
            # with tqdm.tqdm(total=self.population_size) as pbar:
            #     for agent, fitness in pool.starmap(fit, map(lambda a:(pickle.dumps(a),), self.population.values())):
            #         agents_fitness.append((agent, fitness))
            #         print(agents_fitness)
            #         pbar.update()
            agents_fitness = tqdm.tqdm(
                pool.imap_unordered(
                    fit,
                    map(lambda a:pickle.dumps(a), self.population.values()),
                    chunksize=5
                ),
                total=self.population_size
            )
            sorted_agents = sorted(agents_fitness, key=lambda i:i[1], reverse=True)
            best_agents_index = floor(len(sorted_agents)*self.percent_survive)
            best_agents = list(map(lambda i:i[0], sorted_agents[:best_agents_index]))

            ## new_population
            print(f"Keeping top {self.percent_survive}...")
            new_population = best_agents
            def make_new_agents():
                for new_agent_num in range(len(best_agents), self.population_size):
                    ## copy and mutate the best agent mod new agent num
                    new_mutant_agent = (
                        best_agents[(new_agent_num-len(best_agents)) % len(best_agents)]
                        .copy(get_game())
                    )
                    yield (num_mutations, new_agent_num, pickle.dumps(new_mutant_agent))

            print(f"Making new agents...")
            # agents_fitness = tqdm.tqdm(
            #     pool.starmap(agent_mutate, make_new_agents())
            #     total=self.population_size-len(best_agents)
            # )

            new_population += tqdm.tqdm(
                pool.imap_unordered(
                    agent_mutate,
                    make_new_agents(),
                    chunksize=5
                ),
                total=self.population_size-len(best_agents)
            )
            # new_population += pool.starmap(agent_mutate, make_new_agents())

        ## set new pop and return
        self.population = dict(enumerate(new_population))
        return sorted_agents[0]

MUTATE = [
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

aa = Agent(4,get_game()).set_basic()
