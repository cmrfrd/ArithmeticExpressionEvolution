import typing
import collections
import inspect
import functools
import asyncio
import aiosqlite

import nest_asyncio
nest_asyncio.apply()

IteratorType = typing.Union[
typing.Iterator[typing.Any],
typing.AsyncIterator[typing.Any],
typing.Iterable[typing.Any],
typing.AsyncIterable[typing.Any]
]
TransformType = typing.Callable[
[typing.Iterable[typing.Any]],  # *args(input) type
typing.Any  # return type
]

get_loop = lambda: asyncio.get_event_loop()
sync = lambda c: asyncio.get_event_loop().run_until_complete(c)

async def aiter(val):
    if inspect.isawaitable(val):
        val = await val

    if inspect.isasyncgenfunction(val):
        val = val()

    if isinstance(val, collections.abc.AsyncIterator):
        async for v in val:
            yield v
    else:
        for v in iter(val):
            if inspect.isawaitable(v):
                yield (await v)
            else:
                yield v

async def transform_factory(iterator: IteratorType, _type: TransformType=None
                            ) -> typing.Any:
    if not callable(_type):
        raise TypeError('{} is not callable'.format(_type))

    iterator = aiter(iterator)

    if inspect.iscoroutinefunction(_type):
        return await _type(iter([v async for v in iterator]))
    return _type(iter([v async for v in iterator]))

alist = functools.partial(transform_factory, _type=list)

def exhaust(it):
    for _ in it: pass
aexhaust = functools.partial(transform_factory, _type=exhaust)

def gen(it): yield from it
agen = functools.partial(transform_factory, _type=gen)

async def amap(afunc: typing.Callable[[typing.Any], typing.Any],
                iterator: IteratorType) -> typing.AsyncIterator[typing.Any]:
    async for val in aiter(iterator):
        if inspect.iscoroutinefunction(afunc):
            yield await afunc(val)
        else:
            yield afunc(val)


class KV(dict):
    def __init__(self, filename=None, **kwargs):
        self.conn = sync(aiosqlite.connect(filename, **kwargs))
        sync(self.conn.execute("CREATE TABLE IF NOT EXISTS kv (key text unique, value json)"))
        sync(self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_key ON kv (key)"))

    def close(self):
        sync(self.conn.commit())
        sync(self.conn.close())

    async def aiterquery(self, q, *args, **kwargs):
        async with self.conn.execute(q, *args, **kwargs) as cur:
            async for row in cur:
                yield row

    def iterquery(self, q, *args, **kwargs):
        yield from sync(agen(self.aiterquery(q, *args, **kwargs)))

    def __len__(self):
        cur = sync(self.conn.execute('SELECT COUNT(*) FROM kv'))
        rows = sync(cur.fetchone())[0]
        return rows if rows is not None else 0

    def keys(self):
        return list(self.iterkeys())

    def iterkeys(self):
        yield from sync(agen(self.aiterkeys()))

    async def aiterkeys(self):
        async for row in self.aiterquery("SELECT key FROM kv"):
            yield row[0]

    def values(self):
        return list(self.itervalues())

    def itervalues(self):
        yield from sync(agen(self.aitervalues()))

    async def aitervalues(self):
        async for row in self.aiterquery('SELECT value FROM kv'):
            yield row[0]

    def items(self):
        return list(self.iteritems())

    def iteritems(self):
        yield from sync(agen(self.aiteritems()))

    async def aiteritems(self):
        async for row in self.aiterquery('SELECT key, value FROM kv'):
            yield row

    def __contains__(self, key):
        key = str(key)
        return sync(self.acontains(key))

    def contains(self, key):
        return self.__contains__(key)

    async def acontains(self, key):
        key = str(key)
        return (
            await (await self.conn.execute('SELECT 1 FROM kv WHERE key = ?', (key,))).fetchone()
        ) is not None

    def batch_contains(self, keys):
        return sync(agen(self.abatch_contains(keys)))

    async def abatch_contains(self, keys):
        keys = set(list(keys))
        async for k, isin in self.aiterquery(
            f'select key, 1 from kv where key in ({ ",".join("?"*len(keys)) })', tuple(keys)
        ):
            keys.remove(k)
            yield (k, isin)
        for k, isin in map(lambda k: (k, 0), keys):
            yield (k, isin)

    def __getitem__(self, key):
        key = str(key)
        return sync(self.aget(key))

    def get(self, key):
        return self.__getitem__(key)

    async def aget(self, key):
        key = str(key)
        item = await ( await self.conn.execute('SELECT value FROM kv WHERE key = ?', (key,)) ).fetchone()
        if item is None:
            raise KeyError(key)
        return item[0]

    def __setitem__(self, key, value):
        sync(self.aset(key, value))

    def set(self, key, value):
        key = str(key)
        value = str(value)
        sync(self.aset(key, value))

    async def aset(self, key, value):
        key = str(key)
        value = str(value)
        await self.conn.execute('REPLACE INTO kv (key, value) VALUES (?,?)', (key, value))
        await self.conn.commit()

    def batch_set(self, items):
        sync(self.abatch_set(items))

    async def abatch_set(self, items):
        await self.conn.executemany('REPLACE INTO kv (key, value) VALUES (?,?)', [items] if not isinstance(items, list) else items)
        await self.conn.commit()

    def get_with_prefix(self, prefix):
        prefix = str(prefix)
        yield from sync(agen(self.awith_prefix(prefix)))

    async def aget_with_prefix(self, prefix):
        prefix = str(prefix)
        async for row in self.aiterquery('SELECT key FROM kv where key like ?', (prefix+"%",)):
            yield row[0]

    def count_with_prefix(self, prefix):
        prefix = str(prefix)
        return sync(self.acount_with_prefix(prefix))

    async def acount_with_prefix(self, prefix):
        prefix = str(prefix)
        item = await ( await self.conn.execute('SELECT count(key) FROM kv WHERE key like ?', (prefix+"%",)) ).fetchone()
        return item[0]

    def delete_with_prefix(self, prefix):
        prefix = str(prefix)
        yield from sync(agen(self.adelete_with_prefix(prefix)))

    async def adelete_with_prefix(self, prefix):
        prefix = str(prefix)
        await self.conn.execute('DELETE FROM kv WHERE key like ?', (prefix+"%",))
        await self.conn.commit()        

    def __delitem__(self, key):
        key = str(key)
        sync(self.delete(key))

    def delete(self, key):
        self.__delitem__(key)

    async def adelete(self, key):
        key = str(key)
        if not (await self.contains(key)):
            raise KeyError(key)
        await self.conn.execute('DELETE FROM kv WHERE key = ?', (key,))
        await self.conn.commit()

    def __iter__(self):
        return self.iterkeys()
