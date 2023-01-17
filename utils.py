import json
import string
from itertools import islice, chain
from collections import deque
from time import perf_counter
from contextlib import contextmanager
from random import randint, choice, choices, uniform, shuffle, seed
from collections import OrderedDict, Counter
import numpy as np

def rand_n_from_list(n,l):
    working_l = list(l)
    if n > len(working_l):
        ret_l = []
        for i in range(len(working_l)):
            ret_l.append(
                working_l.pop(randint(0,len(working_l)-1))
            )
        return ret_l, []
    else:
        ret_l = []
        for i in range(n):
            ret_l.append(
                working_l.pop(randint(0,len(working_l)-1))
            )
        return ret_l, working_l

def norm_vec(v):
    return v / np.sqrt(np.sum(v**2))

def chunks(items, n):
    items = iter(items)
    for first in items:
        chunk = chain((first,), islice(items, n-1))
        yield chunk
        deque(chunk, 0)

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def get_i(it, index):
    return [elem[index] for elem in it]

def mushed_weighting_fixed(size, mush):
    def distribute(l, i, perc):
        elem = float(l[i])
        l[i] -= elem * perc
        for index in range(i):
            l[index] += (elem*perc)/(i)
        return l
    init = [1/size for i in range(size)]
    for i in range(len(init)-1, 0, -1):
        init = distribute(init, i, mush)
    return init

def mushed_weighting(input: list, mush: float):
    def distribute(l, i, perc):
        elem = float(l[i])
        l[i] -= elem * perc
        for index in range(i):
            l[index] += (elem*perc)/(i)
        return l
    init = input
    for i in range(len(init)-1, 0, -1):
        init = distribute(init, i, mush)
    return init

def print_dict(d):
    print(json.dumps(d, sort_keys=True, indent=4))

def nth(it, n):
    return next(x for i,x in enumerate(it) if i==n)

def uniques(it):
    seen = set()
    for x in it:
        if x not in seen:
            yield x
            seen.add(x)

def iter_size(it):
    c = 0
    for elem in it: c+=1
    return c

def rand_string(size=10):
    return ''.join(choices(string.ascii_lowercase, k=size))

def make_name(l=""):
    return '-'.join(list(map(str,l)) + [rand_string()])

class RDict(OrderedDict):
    def random_key(self):
        i = randint(0, 0 if len(self)==0 else len(self)-1)
        return nth(self.keys(), i)

    def random_value(self):
        return self[self.random_key()]

    def random_item(self):
        k = self.random_key()
        return k, self[k]

    def remove_keys(self, keys):
        for k in keys:
            if k in self:
                del self[k]
