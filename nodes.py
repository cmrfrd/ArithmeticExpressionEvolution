import attrs
from typing import Union, Any
import numpy as np
from functools import reduce
import operator as O

from utils import *

@attrs.define
class Node(object):

    name: str
    ins_range: (int, int)
    outs: int
    ins_atleast: int

@attrs.define
class Constant(Node):

    weights: float = uniform(-1,1)
    ins_range: (int, int) = (0,0)
    outs: int = 1
    ins_atleast: int = 0

    def copy(self, *args):
        return Constant(**attrs.asdict(self))
    def mutate(self):
        self.weights = uniform(-1,1)
        return self
    def __call__(self):
        return self.weights

@attrs.define
class ConstantInt(Node):

    weights: float = randint(0,1)
    ins_range: (int, int) = (0,0)
    outs: int = 1
    ins_atleast: int = 0

    def copy(self, *args):
        return Constant(**attrs.asdict(self))
    def mutate(self):
        self.weights = randint(0,1)
        return self
    def __call__(self):
        return self.weights

@attrs.define
class Op(Node):

    # input_index -> (node_name, node_output_index)
    children: RDict[int, (str, int)] = None
    ins_atleast: int = 1

    def mutate(self):
        return self

    def copy(self, *args, **kwargs):
        return self

@attrs.define
class Noop(Op):
    ins_range: (int, int) = (0,1)
    outs: int = 1

    def copy(self, *args):
        return Noop(**attrs.asdict(self))

MAX_NUM_INS = 4

@attrs.define
class MulOp(Op):
    ins_range: (int, int) = (0,MAX_NUM_INS)
    outs: int = 1

@attrs.define
class DivOp(Op):
    ins_range: (int, int) = (0,MAX_NUM_INS)
    outs: int = 1

@attrs.define
class AddOp(Op):
    ins_range: (int, int) = (0,MAX_NUM_INS)
    outs: int = 1

@attrs.define
class SubOp(Op):
    ins_range: (int, int) = (0,MAX_NUM_INS)
    outs: int = 1

@attrs.define
class MaxOp(Op):
    ins_range: (int, int) = (0,MAX_NUM_INS)
    outs: int = 1

@attrs.define
class MinOp(Op):
    ins_range: (int, int) = (0,MAX_NUM_INS)
    outs: int = 1

@attrs.define
class PowOp(Op):
    ins_range: (int, int) = (0,MAX_NUM_INS)
    outs: int = 1

@attrs.define
class IfLT(Op):
    ins_atleast: int = 2
    ins_range: (int, int) = (0,2)
    outs: int = 1

@attrs.define
class Permute(Op):
    ins_atleast: int = MAX_NUM_INS
    ins_range: (int, int) = (0,MAX_NUM_INS)
    outs: int = MAX_NUM_INS
    weights: Union[None, list] = None

    def mutate(self):
        self.weights = shuffle(range(self.ins_range))
        return self

@attrs.define
class LinearOp(Op):

    weights: Union[None, list] = None
    bias: Union[None, list] = None
    activation_func = "sin"
    ins_atleast: int = None

    def copy(self, *args):
        return LinearOp(**dict(attrs.asdict(self)))

    def reset(self):
        self.ins_atleast: int = self.ins_range[1] - self.ins_range[0]
        self.bias = np.random.uniform(-1, 1, size=self.outs).tolist()
        self.weights = np.random.uniform(-1, 1, size=(self.ins_range[1], self.outs)).tolist()
        return self

    def mutate(self):
        if randint(0,1) == 1:
            np_weights = np.array(self.weights)
            index = tuple(randint(0, s-1) for s in np_weights.shape)
            np_weights[index] = uniform(-1,1)
            self.weights = np_weights.tolist()
        else:
            np_bias = np.array(self.bias)
            index = tuple(randint(0, s-1) for s in np_bias.shape)
            np_bias[index] = uniform(-1,1)
            self.bias = np_bias.tolist()

@attrs.define
class Observable(Node):

    env: Any

    coord_dim: int = 0
    coord: tuple[int] = []

    ins_range: (int, int) = (0,0)
    outs: int = 1
    ins_atleast: int = 0

    def env_shape(self):
        raise NotImplementedError()

    def __call__(self):
        raise NotImplementedError()

    def copy(self, new_env):
        return Observable(
            name=self.name,
            coord_dim=self.coord_dim,
            coord=self.coord,
            ins_range=self.ins_range,
            outs=self.outs,
            ins_atleast=self.ins_atleast,
            env=new_env
        )

    def mutate(self):
        shape: list[int] = self.env_shape()
        self.coord = tuple(randint(0,s-1) for s in shape)
        return self

##
## Matrix observable
##
@attrs.define
class MatrixObservable(Observable):

    coord: tuple[int] = tuple()
    env: Any = None

    ins_range: (int, int) = (0,0)
    outs: int = 1
    ins_atleast: int = 0

    def env_shape(self):
        return self.env.shape

    def __call__(self):
        return float(self.env[tuple(self.coord)])


##
## snake game observable
##

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
class SnakeObservable(Observable):

    max_distance: int = 0
    x: int = 0
    y: int = 0
    z: int = 0
    env: Any = None
    ins_range: (int, int) = (0,0)
    outs: int = 1
    ins_atleast: int = 0

    def env_shape(self):
        return

    def copy(self, new_env):
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
        hx,hy = headAddOp
        d = self.max_distance
        select = grid[hx:hx+2*d+1, hy:hy+2*d+1]
        select_coord = (self.x + d, self.y + d, self.z)

        ## Ensure grid is oriented in the direction of the snakes head before
        ## selecting
        if int(snake._direction) == 0: # north
            return select[select_coord]
        if int(snake._direction) == 1: # east
            return np.rot90(np.rot90(np.rot90(select)))[select_coord]
        if int(snake._direction) == 2: # south
            return np.rot90(np.rot90(select))[select_coord]
        if int(snake._direction) == 3: # west
            return np.rot90(select)[select_coord]

def get_node_class(node_class_str):
    if node_class_str == Noop.__name__:
        return Noop
    elif node_class_str == Constant.__name__:
        return Constant
    elif node_class_str == ConstantInt.__name__:
        return ConstantInt
    elif node_class_str == IfLT.__name__:
        return IfLT
    elif node_class_str == SnakeObservable.__name__:
        return SnakeObservable
    elif node_class_str == MatrixObservable.__name__:
        return MatrixObservable
    elif node_class_str == AddOp.__name__:
        return AddOp
    elif node_class_str == SubOp.__name__:
        return SubOp
    elif node_class_str == MulOp.__name__:
        return MulOp
    elif node_class_str == DivOp.__name__:
        return DivOp
    elif node_class_str == MaxOp.__name__:
        return MaxOp
    elif node_class_str == MinOp.__name__:
        return MinOp
    elif node_class_str == PowOp.__name__:
        return PowOp
    elif node_class_str == LinearOp.__name__:
        return LinearOp
    else:
        raise Exception(f"Unknown node {node_class_str}")

def evaluate(node_name, nodes):
    cache = {}
    def evaluate_helper(search_node_name):
        if search_node_name in cache:
            return cache[search_node_name]
        search_node = nodes[search_node_name]
        if search_node.__class__.__name__ == Constant.__name__:
            result = search_node()
        elif search_node.__class__.__name__ == ConstantInt.__name__:
            result = search_node()
        elif isinstance(search_node, Noop):
            result = [
                evaluate_helper(child_name)
                 for _, (child_name, _) in search_node.children.items()
            ]
        elif isinstance(search_node, Observable):
            result = search_node()
        elif search_node.__class__.__name__ == IfLT.__name__:
            child_a_name, child_b_name = list(child_name for _, (child_name, _) in search_node.children.items())
            result = [
                1 if evaluate_helper(child_a_name) < evaluate_helper(child_b_name) else 0
            ]
        elif search_node.__class__.__name__ == Permute.__name__:
            result = [
                evaluate_helper(search_node.children[i][0])
                for i in search_node.weights
            ]
        elif search_node.__class__.__name__ == AddOp.__name__:
            result = reduce(
                O.add,
                (evaluate_helper(child_name)
                 for _, (child_name, _) in search_node.children.items())
            )
        elif search_node.__class__.__name__ == SubOp.__name__:
            result = reduce(
                O.sub,
                (evaluate_helper(child_name)
                 for _, (child_name, _) in search_node.children.items())
            )
        elif search_node.__class__.__name__ == MulOp.__name__:
            result = reduce(
                O.mul,
                (evaluate_helper(child_name)
                 for _, (child_name, _) in search_node.children.items())
            )
        elif search_node.__class__.__name__ == DivOp.__name__:
            result = reduce(
                lambda a,b: a / b,
                (evaluate_helper(child_name)
                 for _, (child_name, _) in search_node.children.items())
            )
        elif search_node.__class__.__name__ == MaxOp.__name__:
            result = max(
                evaluate_helper(child_name)
                for _, (child_name, _) in search_node.children.items()
            )
        elif search_node.__class__.__name__ == MinOp.__name__:
            result = min(
                evaluate_helper(child_name)
                for _, (child_name, _) in search_node.children.items()
            )
        elif search_node.__class__.__name__ == PowOp.__name__:
            result = reduce(
                lambda a,b: pow(a,b)
                (evaluate_helper(child_name)
                 for _, (child_name, _) in search_node.children.items())
            )
        elif search_node.__class__.__name__ == LinearOp.__name__:
            inputs = np.array(
                [
                 list(
                     evaluate_helper(child_name)
                     for _, (child_name, _) in search_node.children.items()
                 )
                ],
                dtype=np.float32
            )

            weights = np.array(search_node.weights)
            bias = np.array(search_node.bias)
            if search_node.activation_func == "sin":
                result = np.sin(np.dot(inputs, weights) + bias)
            else:
                result = np.dot(inputs, weights) + bias
        else:
            raise Exception(f"Unknown node {search_node.__class__}")
        cache[search_node_name] = result
        return result
    return evaluate_helper(node_name)
