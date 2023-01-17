import attrs
import numpy as np
from copy import deepcopy
import operator as O
from functools import reduce
from random import randint, choice, choices, uniform, shuffle, seed
from typing import Any, Callable
from math import e

from utils import *

CACHE_ENABLED = False
node_eval_cache = {}
def local_cache(func):
    def wrap(ref, *args, **kwargs):
        if ref.name in node_eval_cache:
            return node_eval_cache[ref.name]
        result = func(ref, *args, **kwargs)
        if CACHE_ENABLED:
            node_eval_cache[ref.name] = result
        return result
    return wrap
def clear_cache():
    node_eval_cache = {}

@attrs.define
class Node(object):

    name: str
    ins_range: (int, int)
    outs: int
    ins_atleast: int

@attrs.define
class Random(Node):
    ins_range: (int, int) = (0,0)
    outs: int = 1
    ins_atleast: int = 0

    def copy(self, *args):
        return Random(name=self.name)
    def mutate(self):pass
    def __call__(self):
        return [
            uniform(-1,1)
        ]

@attrs.define
class Constant(Node):

    weights: Any = [ uniform(-1,1) ]
    ins_range: (int, int) = (0,0)
    outs: int = 1
    ins_atleast: int = 0

    def copy(self, *args):
        return Constant(name=self.name,weights=deepcopy(self.weights))
    def mutate(self):
        self.weights[0] += uniform(-1,1)
    def __call__(self):
        return self.weights

def add_ring(arr, val=-1):
    col = val * np.ones((arr.shape[0],1))
    arr = np.concatenate((arr, col), 1)
    arr = np.concatenate((col, arr), 1)
    row = val * np.ones((1, arr.shape[0]+2))
    arr = np.concatenate((row, arr), 0)
    arr = np.concatenate((arr, row), 0)
    return arr

def add_ring_nd(arr, val=-1):
    sur = np.pad(arr, pad_width=1, mode='constant', constant_values=val)
    trim_bord = sur[:,:,1:-1]
    return trim_bord

def add_rings_nd(arr, num, val=0):
    for i in range(num):
        arr = add_ring_nd(arr, val)
    return arr

@attrs.define
class Observable(Node):

    max_distance: int
    x: int
    y: int
    z: int
    env: Any = None
    ins_range: (int, int) = (0,0)
    outs: int = 1
    ins_atleast: int = 0

    def copy(self, new_env=None):
        if not new_env:new_env = self.env
        return Observable(
            name=self.name,
            max_distance=self.max_distance,
            x=self.x,
            y=self.y,
            z=self.z,
            env=new_env
        )

    def mutate(self):
        self.x = randint(-self.max_distance, self.max_distance)
        self.y = randint(-self.max_distance, self.max_distance)
        self.z = randint(0, 3)
        return self

    def __call__(self):
        ## Get data
        snake = self.env.grid.snakes[0]
        head = snake._deque[-1]
        init_grid = self.env.grid.encode()[0]

        ## Encode snake +tail
        grid = (init_grid > 0).astype(np.float32)
        longest = np.prod(grid.shape[:-1])
        snake_index = 1
        for i in range(1, len(snake._deque)+1):
            coord = snake._deque[-i] + (snake_index,)
            grid[coord] = (longest - i + 1) / longest

        ## Add padding around grid
        grid = add_rings_nd(grid, self.max_distance)

        ## Add walls
        walls = add_rings_nd(
            np.zeros(list(init_grid.shape[:-1])+[1]),
            self.max_distance,
            1
        )
        grid = np.concatenate((grid, walls), axis=2)

        ## Slice surrounding head area
        hx,hy = head
        d = self.max_distance
        select = grid[hx:hx+2*d+1, hy:hy+2*d+1]
        select_coord = (self.x + d, self.y + d, self.z)

        ## Ensure grid is oriented in the direction of the snakes head before
        ## selecting
        if int(snake._direction) == 0: # north
            return [ select[select_coord] ]
        if int(snake._direction) == 1: # east
            return [ np.rot90(np.rot90(np.rot90(select)))[select_coord] ]
        if int(snake._direction) == 2: # south
            return [ np.rot90(np.rot90(select))[select_coord] ]
        if int(snake._direction) == 3: # west
            return [ np.rot90(select)[select_coord] ]

@attrs.define
class Op(Node):

    # input_index -> (node_name, node_output_index)
    children: RDict[int, (str, int)]
    nodes: RDict[str, Node] = None
    ins_atleast: int = 1
    activation: Callable[Any, Any] = "sin"


    def mutate(self):
        pass

    def get_node(self, name):
        return self.nodes[name]

    def copy(self, *args):
        return self.__class__(
            name=self.name,
            children=deepcopy(self.children),
        )

    def __call__(self):
        if self.activation == "sin":
            return np.sin(self.call())
        if self.activation == "clamp":
            ret = self.call()
            if ret > 1: return 1
            if ret < -1: return -1
            return ret
        return self.call()

    def call(self):
        raise NotImplementedError()

@attrs.define
class MulOp(Op):
    ins_range: (int, int) = (0,6)
    outs: int = 1
    @local_cache
    def call(self):
        if len(self.children) == 1:
            child_name, index = list(self.children.items())[0][1]
            return [
                self.get_node(child_name)()[index]
            ]
        return [
            reduce(
                O.mul,
                (
                 self.get_node(child_name)()[index]
                 for _, (child_name, index) in self.children.items()
                )
            )
        ]

@attrs.define
class AddOp(Op):
    ins_range: (int, int) = (0,6)
    outs: int = 1
    @local_cache
    def call(self):
        if len(self.children) == 1:
            child_name, index = list(self.children.items())[0][1]
            return [
                self.get_node(child_name)()[index]
            ]
        return [
            reduce(
                O.add,
                (
                 self.get_node(child_name)()[index]
                 for _, (child_name, index) in self.children.items()
                )
            )
        ]

@attrs.define
class MaxOp(Op):
    ins_range: (int, int) = (0,6)
    outs: int = 1
    @local_cache
    def call(self):
        return [
            max(
                self.get_node(child_name)()[index]
                for _, (child_name, index) in self.children.items()
            )
        ]

@attrs.define
class MinOp(Op):
    ins_range: (int, int) = (0,6)
    outs: int = 1
    @local_cache
    def call(self):
        return [
            min(
                self.get_node(child_name)()[index]
                for _, (child_name, index) in self.children.items()
            )
        ]

@attrs.define
class ExpOp(Op):
    ins_range: (int, int) = (0,1)
    outs: int = 1
    @local_cache
    def call(self):
        if len(self.children) > 1:
            raise Exception(f"Too many children for node {self.name}!")

        child_name, index = list(self.children.items())[0][1]
        return [
            e ** self.get_node(child_name)()[index]
        ]

@attrs.define
class FDivOp(Op):
    ins_range: (int, int) = (0,2)
    outs: int = 1
    @local_cache
    def call(self):
        if len(self.children) == 0:
            raise Exception(f"Too few children for node {self.name}!")
        elif len(self.children) == 1:
            child_name, index = list(self.children.items())[0][1]
            return [ self.get_node(child_name)()[index] ]
        elif len(self.children) == 2:
            child1_name, index1 = list(self.children.items())[0][1]
            child2_name, index2 = list(self.children.items())[1][1]
            child1_ans = self.get_node(child1_name)()[index1]
            child2_ans = self.get_node(child2_name)()[index2]
            if child2_ans == 0:
                return [ child1_ans ]
            return [ child1_ans // child2_ans ]
        raise Exception(f"Too many children for node {self.name}!")

@attrs.define
class DivOp(Op):
    ins_range: (int, int) = (0,2)
    outs: int = 1
    @local_cache
    def call(self):
        if len(self.children) == 0:
            raise Exception(f"Too few children for node {self.name}!")
        elif len(self.children) == 1:
            child_name, index = list(self.children.items())[0][1]
            return [ self.get_node(child_name)()[index] ]
        elif len(self.children) == 2:
            child1_name, index1 = list(self.children.items())[0][1]
            child2_name, index2 = list(self.children.items())[1][1]
            child1_ans = self.get_node(child1_name)()[index1]
            child2_ans = self.get_node(child2_name)()[index2]
            if child2_ans == 0:
                return [ child1_ans ]
            return [ child1_ans / child2_ans ]
        raise Exception(f"Too many children for node {self.name}!")

@attrs.define
class ModOp(Op):
    ins_range: (int, int) = (0,2)
    outs: int = 1
    @local_cache
    def call(self):
        if len(self.children) == 0:
            raise Exception(f"Too few children for node {self.name}!")
        elif len(self.children) == 1:
            child_name, index = list(self.children.items())[0][1]
            return [ self.get_node(child_name)()[index] ]
        elif len(self.children) == 2:
            child1_name, index1 = list(self.children.items())[0][1]
            child2_name, index2 = list(self.children.items())[1][1]
            child1_ans = self.get_node(child1_name)()[index1]
            child2_ans = self.get_node(child2_name)()[index2]
            if child2_ans == 0:
                return [ child1_ans ]
            return [ child1_ans % child2_ans ]
        raise Exception(f"Too many children for node {self.name}!")

@attrs.define
class LinearOp(Op):

    weights: Any = None
    bias: Any = None
    # activation_func: Callable[Any, Any] = lambda x: np.tanh(x)
    activation_func: Callable[Any, Any] = "sin"
    ins_atleast: int = None

    def copy(self, *args):
        return LinearOp(
            name=self.name,
            weights=self.weights,
            bias=self.bias,
            ins_range=self.ins_range,
            outs=self.outs,
            ins_atleast=self.ins_range[1] - self.ins_range[0],
            children=deepcopy(self.children),
        )

    def reset(self):
        self.ins_atleast: int = self.ins_range[1] - self.ins_range[0]
        self.bias = np.random.uniform(-1, 1, size=self.outs)
        self.weights = np.random.uniform(-1, 1, size=(self.ins_range[1], self.outs))
        return self

    def mutate(self):
        if randint(0,1) == 1:
            index = tuple(randint(0, s-1) for s in self.weights.shape)
            self.weights[index] += uniform(-1,1)
        else:
            index = tuple(randint(0, s-1) for s in self.bias.shape)
            self.bias[index] += uniform(-1,1)

    def __call__(self):
        msg = f"Number of children of node {self.name} must match it's number of inputs"
        assert len(self.children) == self.ins_range[1]-self.ins_range[0], msg

        ## Evaluate inputs
        #inputs = np.empty((1,self.ins_range[0]), dtype=np.float32)
        #for i in range(inputs.shape[1]):
        #    inputs[0,i] = self.children[i]()
        inputs = np.array(
            [
             [
              self.get_node(child_name)()[child_out]
              for child_num,(child_name,child_out) in self.children.items()
             ]
            ],
            dtype=np.float32
        )

        if self.activation_func == "sin":
            return (
                np.sin(
                    np.dot(inputs, self.weights) + self.bias
                )
            )
        return (
            np.dot(inputs, self.weights) + self.bias
        )
