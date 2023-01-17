import string
from collections import OrderedDict
from random import randint, choice, choices, uniform, shuffle, seed

class RDict(OrderedDict):
    def __str__(self):
        return f"<RDict size={len(self)}>"

    def __repr__(self):
        return f"<RDict size={len(self)}>"

    def random_key(self):
        i = randint(0, 0 if len(self)==0 else len(self)-1)
        return nth(self.keys(), i)

    def random_value(self):
        """ Return a random value from this dictionary in O(1) time """
        return self[self.random_key()]

    def random_item(self):
        """ Return a random key-value pair from this dictionary in O(1) time """
        k = self.random_key()
        return k, self[k]

    def remove_keys(self, keys):
        for k in keys:
            if k in self:
                del self[k]


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

def make_name():
    return ''.join(choices(string.ascii_lowercase, k=10))
