import blake3
import json
import re
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from random import sample
from networkx.drawing.nx_agraph import graphviz_layout
from itertools import product

from nodes import *
from utils import *

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

def serialize(agent):
    serialized_nodes = {}
    for node_name, node in agent.nodes.items():
       serialzied_node_data = attrs.asdict(node)
       if hasattr(node, "env"): serialzied_node_data['env'] = None
       serialized_nodes[node_name] = {
           "type": node.__class__.__name__,
           **serialzied_node_data
       }

    return json.dumps({
        "node_domain": agent.node_domain,
        "mutate_domain": agent.mutate_domain,
        "nodes": serialized_nodes
    })

def deserialize(s, env=None):
    data = json.loads(s)
    agent = Agent(node_domain=data["node_domain"], mutate_domain=data["mutate_domain"], env=env)
    for node_name, node_data in data['nodes'].items():
        type_str = node_data.pop("type")
        type_class = get_node_class(type_str)
        if hasattr(type_class, "env"): node_data['env'] = env
        agent.nodes[node_name] = type_class(**node_data)
        if hasattr(type_class, "children"):
            agent.nodes[node_name].children = RDict(
                **dict(
                    (int(i),(c,int(o)))
                    for i,(c,o) in agent.nodes[node_name].children.items()
                )
            )
    agent.head = ('head', agent.nodes['head'])
    return agent

def dfs(node_name, nodes, return_path=False, return_terminal_paths=False, additional_graph=None, repeats=False):
    seen = set()
    if not additional_graph:
        additional_graph = RDict()
    def dfs_helper(search_node_name, path=None):
        if not path:
            path = []
        search_node = nodes[search_node_name]
        path.append(search_node_name)
        if return_path: (yield path)
        if (not return_path) and (not return_terminal_paths): (yield path[-1])
        if (not repeats) and (search_node_name not in seen): seen.add(search_node_name)
        if hasattr(search_node, "children"):
            for _, (child_name, _) in sorted(search_node.children.items(), key=lambda el: el[0]):
                yield from dfs_helper(child_name, path)
            for child_name in additional_graph.get(search_node_name, []):
                yield from dfs_helper(child_name, path)
        else:
            if return_terminal_paths: yield (path)
        path.pop()
    yield from dfs_helper(node_name)

class Agent:
    def __init__(self, node_domain, mutate_domain, env=None):
        self.node_domain = node_domain
        self.mutate_domain = mutate_domain
        self.nodes: dict[str, Node] = RDict()
        self.head: tuple(str, Node) = ("head", None)
        self.env = env

    def set_env(self, new_env):
        for node_name, node in self.nodes.items():
            if isinstance(node, Observable):
                node: Observable = node
                node.env = new_env
        self.env = new_env
        return self

    def fuzz(self, mutations=10000):
        for _ in range(int(mutations)):
            self.mutate()
            print(self())
        return self

    def copy(self, new_env=None):
        ## Create new agent and fill nodes
        new_agent = Agent(self.node_domain, self.mutate_domain, new_env if new_env else self.env)
        for node_name, node in self.nodes.items():
            new_agent.nodes[node_name] = deepcopy(node)
        ## Set children of nodes
        for node_name, node in self.nodes.items():
            if hasattr(node, 'children'):
                for input_num, (child_node_name, output_num) in node.children.items():
                    new_agent.nodes[node_name].children[input_num] = \
                      (child_node_name, output_num)
        new_agent.head = ('head', new_agent.nodes['head'])
        return new_agent

    def to_networkx(self):
        G = nx.DiGraph()
        for node_name, node in self.nodes.items():
            G.add_node(node_name)
            if hasattr(node, 'children'):
                for _,(c,_) in self.nodes[node_name].children.items():
                    G.add_node(c)
                    G.add_edge(node_name, c)
        return G

    def get_hash(self):
        h = blake3.blake3("".encode())
        for node_name in dfs("head", self.nodes, repeats=True):
            h.update(self.nodes[node_name].__class__.__name__.encode())
        return int.from_bytes(h.digest(), 'big')

    def set_basic(self, out_dim):
        nonop_node_classes = list(filter(lambda ncs: not hasattr(get_node_class(ncs[0]), "children"), self.node_domain))
        if not nonop_node_classes: raise Exception("Must be atleast one terminal node class")
        nonop_node_class = nonop_node_classes[randint(0, len(nonop_node_classes)-1)][0]

        name = make_name()
        nodes = [
            Noop(
                name="head",
                ins_range=(0,out_dim),
                outs=out_dim,
                children=RDict(**{
                    0:(name,0),
                })
            ),
            get_node_class(nonop_node_class)(name=name),
        ]
        self.nodes = RDict()
        self.add_nodes(nodes)
        return self

    def noop(self):
        pass

    def stats(self):
        total_nodes = 0
        total_constants = 0
        total_nonconstants = 0
        total_edges = 0
        total_unused_inputs = 0
        total_capacity = 0
        for node_name, n in self.nodes.items():
            total_nodes += 1
            if hasattr(n, "children"):
                total_edges += len(n.children)
                total_unused_inputs += n.ins_range[1] - n.ins_range[0] - len(n.children)
                total_nonconstants += 1
            else:
                total_edges += 0
                total_constants += 1
            total_capacity += n.ins_range[1] - n.ins_range[0]
            if n.__class__ == Constant: total_constants += 1

        all_node_types_str = list(set(uniques(v.__class__.__name__ for v in self.nodes.values())).union(set(n for n,_ in self.node_domain)))

        all_nodes_counts = dict((str(n), 0) for n in all_node_types_str)
        for node_name, node in self.nodes.items(): all_nodes_counts[node.__class__.__name__] += 1

        all_pairs_counts = dict((str(tuple([a,b])), 0) for a,b in product(all_node_types_str,repeat=2))
        paths = list(list(p) for p in dfs("head", self.nodes, return_terminal_paths=True))
        pairs = uniques(chain(*(window(path, 2) for path in paths)))
        pairs_node_type = (tuple([self.nodes[a].__class__.__name__, self.nodes[b].__class__.__name__]) for a,b in pairs)
        for pair in pairs_node_type: all_pairs_counts[str(pair)] += 1

        res = dict(
            total_nodes=total_nodes,
            total_constants=total_constants,
            total_nonconstants=total_nonconstants,
            total_capacity=total_capacity,
            current_capacity=round(total_edges / total_capacity, 3),
            total_edges=total_edges,
            total_unused_inputs=total_unused_inputs,
            deepest=max(map(len, dfs("head", self.nodes, return_path=True))),
            floating_nodes=total_nodes - len(set(dfs("head", self.nodes))),
            **all_nodes_counts,
            **all_pairs_counts,
        )
        return res

    def add_nodes(self, nodes):
        assert "head" in map(lambda n:n.name, nodes), "One node must be the head"

        for n in nodes:
            if n.name == "head":
                self.head = ("head", n)
            self.nodes[n.name] = n

    def get_children_names(self, node_names: list[str]):
        seen = set()
        for node_name in node_names:
            if node_name in seen: continue # Skip if we've already seen the child
            search_name_iter = dfs(node_name, self.nodes, repeats=False)
            for search_node_name in search_name_iter:
                if search_node_name not in seen:
                    yield search_node_name
                    seen.add(search_node_name)

    def get_random_node_name(self):
        node_names = list(self.nodes.keys())
        node_names.remove("head")
        return node_names[randint(0, len(node_names)-1)]

    def get_parent_names(self, node_names: list[str], reassign_dict: dict[str, list] = None):
        node_names = set(node_names)
        parents_unique_names = set()
        for search_node_name, search_node in self.nodes.items():
            for path in dfs(search_node_name, self.nodes, return_path=True, additional_graph=reassign_dict):
                ## found path to a target node
                if path[-1] in node_names:
                    ## yield all the new paths nodes names and add them to the set
                    new_parents_unique_names = set(path) - parents_unique_names
                    yield from new_parents_unique_names
                    parents_unique_names.update(new_parents_unique_names)

    def get_refs_of_node_name(self, node_name):
        '''Yield nodes whose child is "node"'''
        for search_node_name, search_node in self.nodes.items():
            if hasattr(search_node, "children"):
                for child_num, (child_name, out) in search_node.children.items():
                    if node_name == child_name:
                        yield (search_node, child_num)

    def get_random_output_name(self, avoid=None, include_head=False):
        if not avoid:
            avoid = []

        ## Copy nodes, and remove any we want to avoid
        select_nodes = deepcopy(self.nodes)
        select_nodes.pop("head") if not include_head else None
        select_nodes.remove_keys(avoid)

        ## Count the total number of "outs" and return none if there are none
        total_num_outs = sum(node.outs for name, node in select_nodes.items())
        if total_num_outs == 0: return (None, None)

        ## Choose a random "out", countdown, return node from original set
        rand_out = randint(1, total_num_outs)
        out_node_name = None
        for node_name, node in select_nodes.items():
            if rand_out - (node.outs) <= 0:
                out_node_name = node_name
                break
            rand_out -= node.outs
        return out_node_name, rand_out-1

    def get_random_input_name(self, avoid=None, include_head=False):
        if not avoid:
            avoid = []

        ## Copy nodes, and remove any we want to avoid
        select_nodes = deepcopy(self.nodes)
        select_nodes.pop("head") if not include_head else None
        select_nodes.remove_keys(avoid)

        ## count the total number of "inputs"
        total_num_ins = sum(node.ins_range[1] - node.ins_range[0] for name, node in select_nodes.items())
        if total_num_ins == 0: return (None, None)

        rand_in = randint(1, total_num_ins)
        in_node_name = None
        for node_name, node in select_nodes.items():
            if rand_in - (node.ins_range[1] - node.ins_range[0]) <= 0:
                in_node_name = node_name
                break
            rand_in -= (node.ins_range[1] - node.ins_range[0])
        return in_node_name, rand_in-1

    def pop_node(self, node_name):
        pass

    def get_random_child(self, avoid=None):
        if not avoid:
            avoid = []

        ## Copy nodes, and remove any we want to avoid
        select_nodes = deepcopy(self.nodes)
        select_nodes.pop("head")
        select_nodes.remove_keys(avoid)

        ## count all the little children
        total_num_children = sum(
            len(node.children) if hasattr(node, "children") else 0
            for node_name, node in select_nodes.items()
        )
        if total_num_children == 0: return (None, None)

        ## get some child
        some_child_num = randint(0, total_num_children-1) if total_num_children > 1 else 0

        ## get that some child and return it
        def child_iter():
            for node_name, node in select_nodes.items():
                if hasattr(node, "children"):
                    for child_num, (child_name, out) in node.children.items():
                        yield (node_name, child_num)
        return nth(child_iter(), some_child_num)

    def make_new_node(self, only_children=False):
        ## init random node with no children
        nd = []
        if only_children:
            nd = list(filter(lambda el:hasattr(get_node_class(el[0]),"children"),self.node_domain))
            nd = [(n,1/len(nd)) for i,(n,p) in enumerate(nd)]
        else:
            nd = self.node_domain
        node_class_str = np.random.choice([n for n,p in nd], p=[p for n,p in nd])
        node_class = get_node_class(node_class_str)
        if hasattr(node_class,"children"):
            new_node = node_class(
                name=make_name([node_class.__name__]),
                children=RDict()
            )
        elif issubclass(node_class, Observable):
            new_node = node_class(
                name=make_name([node_class.__name__]),
                env=self.env
            )
        else:
            new_node = node_class(
                name=make_name([node_class.__name__])
            )
        return new_node

    def mutate(self):
        try:
            mutation_op = np.random.choice([n for n,p in self.mutate_domain], p=[p for n,p in self.mutate_domain])
            getattr(self, mutation_op)()
        except Exception as e:
            print(e, mutation_op)
            raise e
        return self

    def mutate_n(self, n):
        for _ in range(int(n)):
            self.mutate()
        return self

    def update_random_node_type(self):
        all_node_names = list(self.nodes.keys())
        all_node_names.remove("head")

        all_node_names_only_children = list(
            filter(
                lambda node_name: hasattr(self.nodes[node_name], "children"),
                all_node_names,
            )
        )
        if not all_node_names_only_children: return
        random_node_name = all_node_names_only_children[randint(0,len(all_node_names_only_children)-1)]
        random_node = self.nodes[random_node_name]

        ## choose random node type, add children until it can't take any more
        new_node = self.make_new_node(only_children=True)
        for i,(n,o) in random_node.children.items():
            if len(new_node.children) < (new_node.ins_range[1] - new_node.ins_range[0]):
                new_node.children[i] = (n,o)

        ## change all the references of the random node to the new node
        for ref_node, out_num in self.get_refs_of_node_name(random_node_name):
            _,o = self.nodes[ref_node.name].children[out_num]
            self.nodes[ref_node.name].children[out_num] = (new_node.name,o)

        # add the new node
        self.nodes[new_node.name] = new_node

        ## delete the old node
        del self.nodes[random_node_name]

        return self

    def add_random_node(self):

        ## make a new node
        new_node = self.make_new_node()

        ## catch any quick errors and mutate
        if new_node.name in self.nodes:
            raise Exception("Node name already exists!")
        new_node.mutate()

        ## Find the max number of inputs to our "new" node
        ## then "wire" them in if any
        new_node_num_inputs = new_node.ins_range[1] - new_node.ins_range[0]
        if new_node_num_inputs > 0:
            ## choose a random num of outputs to wire
            num_outputs_to_wire = randint(new_node.ins_atleast, new_node_num_inputs)

            ## wire some random outputs into the "new" node
            for out in range(num_outputs_to_wire):
                out_node_name, num_out = self.get_random_output_name()
                new_node.children[out] = (out_node_name, num_out)

        ## Take the output(s) of this node and feed them in the inputs of some
        ## others
        if not hasattr(new_node, "children"): avoid = None
        else: avoid = set(n for n in self.get_children_names([c for _,(c,_) in new_node.children.items()]))

        num_outputs_to_wire = randint(1, new_node.outs)
        for out in range(num_outputs_to_wire):

            ## find an input that isn't a child of this "new node"
            ## and if all nodes are saturated, just exit
            in_node_name, num_in = self.get_random_input_name(avoid=avoid)
            if not in_node_name: break

            ## make the wire
            self.nodes[in_node_name].children[num_in] = (new_node.name, out)

        ## Add the new node
        self.nodes[new_node.name] = new_node
        return self

    def delete_random_node_and_edges(self, node_name=None):
        if not node_name:
            select_nodes = deepcopy(self.nodes)
            select_nodes.pop('head')
            tobe_deleted_node_name, tobe_deleted_node = select_nodes.random_item()
        elif node_name in self.nodes:
            tobe_deleted_node_name, tobe_deleted_node = node_name, self.nodes[node_name]
        else:
            raise Exception("Bad node name to delete")

        ## stop us from deleting the only terminal node
        if (tobe_deleted_node.ins_atleast == 0) and iter_size(filter(lambda n:n.ins_atleast == 0, self.nodes.values())) == 1:
            # raise Exception("Dag needs atleast one terminal node")
            return

        ## search for references that need to be "deleted" or "reassigned"
        delete_actions = []
        reassign_actions = [] # (search_node, inp_num, (new_child, out))
        for ref_node, ref_node_child_num in self.get_refs_of_node_name(tobe_deleted_node_name):

            ## If there are enough children in this reference node to operate,
            ## then it's safe to delete the edge
            ref_node_deletes = iter_size(filter(lambda item: item[0] == ref_node, delete_actions))
            if (len(ref_node.children) - ref_node_deletes) > ref_node.ins_atleast:
                delete_actions.append((ref_node, ref_node_child_num))
                continue

            ## If the to be deleted node is *not* terminal, reroute the
            ## inputs from the to be deleted node, into this referenced one
            ## but check that the to be deleted not is not a parent of the ref node
            if tobe_deleted_node.ins_atleast > 0:
                did_reassign = False
                for inp, (new_child_name, out) in sample(list(tobe_deleted_node.children.items()), len(tobe_deleted_node.children)):
                    if ref_node.name not in self.get_parent_names([tobe_deleted_node.name]):
                        reassign_actions.append((ref_node, ref_node_child_num, (new_child_name, out)))
                        did_reassign = True
                        break
                if did_reassign:
                    continue

            ## the important edge case ...
            ## 1. The to be deleted node *is* terminal
            ## 2. the reference node edge can't be deleted, it *must* be rerouted to some output
            ## <- get the nodes who need the "reference node" to be evaluated
            ##    including those that will potentially be reassigned
            reassign_dict = RDict()
            for reassign_ref_node, reassign_ref_node_child_num, (new_child_name, out) in reassign_actions:
                if reassign_ref_node.name not in reassign_dict:
                    reassign_dict[reassign_ref_node.name] = [new_child_name]
                else:
                    reassign_dict[reassign_ref_node.name].append(new_child_name)
            recently_reassigned_nodes_whose_children_contain_current_ref = uniques(
                map(
                    lambda item: item[0].name,
                    filter(
                        lambda item: (
                            ref_node.name in dfs(
                                item[2][0],
                                self.nodes,
                                additional_graph=reassign_dict
                            )
                        ),
                        reassign_actions
                    )
                )
            )

            ## Get all the parents of these recently reassigned nodes
            node_names_recently_reassigned_to_current_ref = \
              map(lambda item: item[0].name,filter(lambda item: item[2][0] == ref_node.name, reassign_actions))
            parents_of_nodes_recently_reassigned_to_current_ref = uniques(
                parent_name
                for parent_name in self.get_parent_names(list(node_names_recently_reassigned_to_current_ref), reassign_dict=reassign_dict)
            )

            ## Get the parents of the curent ref node
            ref_node_parent_names = uniques(self.get_parent_names([ref_node.name], reassign_dict=reassign_dict))

            influenced_nodes_names = chain(
                [tobe_deleted_node_name],
                recently_reassigned_nodes_whose_children_contain_current_ref,
                ref_node_parent_names,
                parents_of_nodes_recently_reassigned_to_current_ref,
            )

            ## Try to find a random output the "search node" can "hook" into
            out_node, num_out = self.get_random_output_name(avoid=influenced_nodes_names)
            if not out_node:
                return

            ## Reassign the ref node edge
            reassign_actions.append((ref_node, ref_node_child_num, (out_node, num_out)))
            continue

        ## run all the delete actions
        for n, d_inp_num in delete_actions:
            del n.children[d_inp_num]

        ## run all the reassign actinos
        for n, r_inp_num, (out_node, num_out) in reassign_actions:
            n.children[r_inp_num] = (out_node, num_out)

        ## we can now delete the node
        del self.nodes[tobe_deleted_node_name]
        return self

    def add_random_edge(self):
        out_node_name, num_out = self.get_random_output_name()
        in_node_name, num_in = self.get_random_input_name(avoid=self.get_children_names([out_node_name]))
        if not in_node_name:
            return
        self.nodes[in_node_name].children[num_in] = (out_node_name, num_out)
        return self

    def update_random_edge(self):
        out_node_name, num_out = self.get_random_output_name()

        ## Get some child that isn't a child of the random output node
        in_node_name, num_in = self.get_random_child(avoid=self.get_children_names([out_node_name]))
        if not in_node_name:
            return

        ## update the edge
        self.nodes[in_node_name].children[num_in] = (out_node_name, num_out)
        return self

    def update_head_edges(self):
        for inp, (child_name, out) in self.nodes["head"].children.items():
            self.nodes["head"].children[inp] = self.get_random_output_name()
        return self

    def delete_random_edge(self):
        nodes_names_with_spare_edges = []
        for node_name, node in self.nodes.items():
            ## ignore constants
            if (node.ins_range[1] - node.ins_range[0]) == 0: continue

            ## if there are spare children, any of them are fair game
            if len(node.children) > node.ins_atleast:
                nodes_names_with_spare_edges += [
                    (node_name, child_id) for child_id in node.children.keys()
                ]

        ## end if there are no spare edges
        if len(nodes_names_with_spare_edges) == 0: return False

        ## choose the edge and delete it
        n_name, child_id_to_delete = choice(nodes_names_with_spare_edges)
        del self.nodes[n_name].children[child_id_to_delete]
        return self

    def update_weight(self):
        has_weights = lambda k: hasattr(self.nodes[k],"weights") or hasattr(self.nodes[k],"coord")
        weight_nodes_names = list(filter(has_weights,self.nodes.keys()))
        if weight_nodes_names:
            self.nodes[choice(weight_nodes_names)].mutate()
        return self

    def prune(self):
        new_nodes = RDict()
        for node_name in dfs("head", self.nodes):
            new_nodes[node_name] = self.nodes[node_name]
        self.nodes = new_nodes
        return self

    def simplify(self):
        def simplify_helper():
            for node_name in dfs("head", self.nodes):
                if hasattr(self.nodes[node_name], "children"):
                    if len(self.nodes[node_name].children) == 1:
                        i,(node_child_name,o) = list(self.nodes[node_name].children.items())[0]
                        if hasattr(self.nodes[node_child_name], "children"):
                            if len(self.nodes[node_child_name].children) == 1:
                                # "node_child_name" is a noop node
                                ii,(node_child_child_name,oo) = list(self.nodes[node_child_name].children.items())[0]
                                self.nodes[node_name].children[i] = \
                                    (node_child_child_name,oo)
                                del self.nodes[node_child_name]
            return self
        h = self.get_hash()
        h_p = simplify_helper().get_hash()
        while h != h_p:
            h = h_p
            h_p = simplify_helper().get_hash()
        return self

    def __call__(self):
        return evaluate("head", self.nodes)
