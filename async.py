def result_noraise(future, flat=True):
    try:
        res = future.result()
        return res if flat else (res, None)
    except BaseException as exc:
        return exc if flat else (None, exc)

class getres:
    dont = lambda fut: fut
    flat = partial(result_noraise, flat=True)
    pair = partial(result_noraise, flat=False)

async def iterwait(futures, *, flat=True, get_result=getres.flat,
                   timeout=None, yield_when=aio.ALL_COMPLETED):
    _futures = futures[:]
    while _futures:
        done, _futures = await aio.wait(_futures, timeout=timeout,
                                        return_when=yield_when)
        if flat:
            for fut in done:
                yield get_result(fut)
        else:
            yield [get_result(fut) for fut in done]

class AioPool(object):

    def __init__(self, size=1024, *, loop=None):

        self.loop = loop or _get_loop()

        self.size = size
        self._executed = 0
        self._joined = set()
        self._waiting = {}  # future -> task
        self._active = {}  # future -> task
        self.semaphore = aio.Semaphore(value=self.size)

    async def itermap(self, fn, iterable, cb=None, ctx=None, *, flat=True,
                      get_result=getres.flat, timeout=None,
                      yield_when=aio.ALL_COMPLETED):
        '''Spawns coroutines created with `fn` for each item in `iterable`, then
        waits for results with `iterwait`. See docs for `map_n` and `iterwait`.
        '''
        futures = self.map_n(fn, iterable, cb, ctx)
        generator = iterwait(futures, flat=flat, timeout=timeout,
                    get_result=get_result, yield_when=yield_when)
        async for batch in generator:
            yield batch  # TODO is it possible to return a generator?

    async def iterrun(self, iterable, cb=None, ctx=None, *, flat=True,
                get_result=getres.flat, timeout=None,
                yield_when=aio.ALL_COMPLETED):
        '''Spawns coroutines created with `fn` for each item in `iterable`, then
        waits for results with `iterwait`. See docs for `map_n` and `iterwait`.
        '''
        futures = self.run_n(iterable, cb, ctx)
        generator = iterwait(futures, flat=flat, timeout=timeout,
                    get_result=get_result, yield_when=yield_when)
        async for batch in generator:
            yield batch  # TODO is it possible to return a generator?

    async def __aenter__(self):
          return self

    async def __aexit__(self, ext_type, exc, tb):
          await self.join()

    def __len__(self):
          return len(self._waiting) + self.n_active

    @property
    def n_active(self):
          '''Counts active coroutines'''
          return self.size - self.semaphore._value

    @property
    def is_empty(self):
          '''Returns `True` if no coroutines are active or waiting.'''
          return 0 == len(self._waiting) == self.n_active

    @property
    def is_full(self):
          '''Returns `True` if `size` coroutines are already active.'''
          return self.size <= len(self)

    async def join(self):
        '''Waits (blocks) for all spawned coroutines to finish, both active and
        waiting. *Do not `join` inside spawned coroutine*.'''

        if self.is_empty:
            return True

        fut = self.loop.create_future()
        self._joined.add(fut)
        try:
            return await fut
        finally:
            self._joined.remove(fut)

    def _release_joined(self):
        if not self.is_empty:
            raise RuntimeError()  # TODO better message

        for fut in self._joined:
            if not fut.done():
                fut.set_result(True)

    def _build_callback(self, cb, res, err=None, ctx=None):
        # not sure if this is a safe code( in case any error:
        # return cb(res, err, ctx), None

        bad_cb = RuntimeError('cb should accept at least one argument')
        to_pass = (res, err, ctx)

        nargs = cb.__code__.co_argcount
        if nargs == 0:
            return None, bad_cb

        # trusting user here, better ideas?
        if cb.__code__.co_varnames[0] in ('self', 'cls'):
            nargs -= 1  # class/instance method, skip first arg

        if nargs == 0:
            return None, bad_cb

        try:
            return cb(*to_pass[:nargs]), None
        except Exception as e:
            return None, e

    async def _wrap(self, coro, future, cb=None, ctx=None):
        res, exc, tb = None, None, None
        try:
            res = await coro
        except BaseException as _exc:
            exc = _exc
            tb = traceback.format_exc()
        finally:
            self._executed += 1

        while cb:
            err = None if exc is None else (exc, tb)

            _cb, _cb_err = self._build_callback(cb, res, err, ctx)
            if _cb_err is not None:
                exc = _cb_err  # pass to future
                break

            wrapped = self._wrap(_cb, future)
            self.loop.create_task(wrapped)
            return

        self.semaphore.release()

        if not future.done():
            if exc:
                future.set_exception(exc)
            else:
                future.set_result(res)

        del self._active[future]
        if self.is_empty:
            self._release_joined()

    async def _spawn(self, future, coro, cb=None, ctx=None):
          acq_error = False
          try:
              await self.semaphore.acquire()
          except BaseException as e:
              acq_error = True
              coro.close()
              if not future.done():
                  future.set_exception(e)
          finally:
              del self._waiting[future]

          if future.done():
              if not acq_error and future.cancelled():  # outside action
                  self.semaphore.release()
          else:  # all good, can spawn now
              wrapped = self._wrap(coro, future, cb=cb, ctx=ctx)
              task = self.loop.create_task(wrapped)
              self._active[future] = task
          return future

    async def spawn(self, coro, cb=None, ctx=None):
          '''Waits for pool space and creates task for given `coro` coroutine,
          returns a future for it's result.
          If callback `cb` coroutine function (not coroutine itself!) is passed,
          `coro` result won't be assigned to created future, instead, `cb` will
          be executed with it as a first positional argument. Callback function
          should accept 1,2 or 3 positional arguments. Full callback sigature is
          `cb(res, err, ctx)`. It makes no sense to create a callback without
          `coro` result, so first positional argument is mandatory.
          Second positional argument of callback will be error, which `is None`
          if coroutine did not crash and wasn't cancelled. In case any exception
          was raised during `coro` execution, error will be a tuple containing
          (`exc` exception object, `tb` traceback string). if you wish to ignore
          errors, you can pass callback without seconds and third positional
          arguments.
          If context `ctx` is passed to `spawn`, it will be re-sent to callback
          as third argument. If you don't plan to use any context, you can create
          callback with positional arguments only for result and error.
          '''
          future = self.loop.create_future()
          self._waiting[future] = self.loop.create_future()  # as a placeholder
          return await self._spawn(future, coro, cb=cb, ctx=ctx)

    def spawn_n(self, coro, cb=None, ctx=None):
          '''Creates waiting task for given `coro` regardless of pool space. If
          pool is not full, this task will be executed very soon. Main difference
          is that `spawn_n` does not block and returns future very quickly.
          Read more about callbacks in `spawn` docstring.
          '''
          future = self.loop.create_future()
          task = self.loop.create_task(self._spawn(future, coro, cb=cb, ctx=ctx))
          self._waiting[future] = task
          return future

    async def exec(self, coro, cb=None, ctx=None):
          '''Waits for pool space, then waits for `coro` (and it's callback if
          passed) to finish, returning result of `coro` or callback (if passed),
          or raising error if smth crashed in process or was cancelled.
          Read more about callbacks in `spawn` docstring.
          '''
          return await (await self.spawn(coro, cb, ctx))

    def run_n(self, coro_iterable, cb=None, ctx=None):
          futures = []
          for coro in coro_iterable:
              fut = self.spawn_n(coro, cb, ctx)
              futures.append(fut)
          return futures

    def map_n(self, fn, iterable, cb=None, ctx=None):
          '''Creates coroutine with `fn` function for each item in `iterable`,
          spawns each of them with `spawn_n`, returning futures.
          Read more about callbacks in `spawn` docstring.
          '''
          futures = []
          for it in iterable:
              fut = self.spawn_n(fn(it), cb, ctx)
              futures.append(fut)
          return futures

    async def map(self, fn, iterable, cb=None, ctx=None, *,
              get_result=getres.flat):
          '''Spawns coroutines, created with `fn` function for each item in
          `iterable`, waits for all of them to finish, crash or be cancelled,
          returning resuls.
          `get_result` is function, that accepts future as only positional
          argument, whose goal is to extract result from future. You can pass
          your own, or use inluded `getres` object, that has 3 extractors:
          `getres.dont` will return future untouched, `getres.flat` will return
          exception object if coroutine crashed or was cancelled, otherwise will
          return result of a coroutine (or of the callback), `getres.pair` will
          return tuple of (`result', 'exception` object) with None in place of
          missing item.
          Read more about callbacks in `spawn` docstring.
          '''
          futures = []
          for it in iterable:
              fut = await self.spawn(fn(it), cb, ctx)
              futures.append(fut)

          await aio.wait(futures)
          return [get_result(fut) for fut in futures]

    async def cancel(self, *futures, get_result=getres.flat):
          '''Cancels spawned or waiting tasks, found by their `futures`. If no
          `futures` are passed -- cancels all spwaned and waiting tasks.
          Cancelling futures, returned by pool methods, usually won't help you
          to cancel executing tasks, so you have to use this method.
          Returns tuple of (`cancelled` count of cancelled tasks, `results`
          collected from futures of cancelled tasks).
          '''
          tasks, _futures = [], []

          if not futures:  # meaning cancel all
              tasks.extend(self._waiting.values())
              tasks.extend(self._active.values())
              _futures.extend(self._waiting.keys())
              _futures.extend(self._active.keys())
          else:
              for fut in futures:
                  task = self._active.get(fut, self._waiting.get(fut))
                  if task:
                      tasks.append(task)
                      _futures.append(fut)

          cancelled = 0
          if tasks:
              cancelled = sum(1 for task in tasks if task.cancel())
              await aio.wait(tasks)  # let them actually cancel
          # need to collect them anyway, to supress warnings
          return cancelled, [get_result(fut) for fut in _futures]


def _get_loop():
    return aio.get_event_loop()

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

    if isinstance(val, typing.AsyncIterator):
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

async def exaust(iterator: IteratorType):
    async for _ in aiter(iterator):
        pass

async def amap(afunc: typing.Callable[[typing.Any], typing.Any],
               iterator: IteratorType) -> typing.AsyncIterator[typing.Any]:
    async for val in aiter(iterator):
        if inspect.iscoroutinefunction(afunc):
            yield await afunc(val)
        else:
            yield afunc(val)


async def pool_run(loop, coro_iter, size, batch_size=512):
    pool = AioPool(size=size, loop=loop)
    for coro_batch in chunks(coro_iter, batch_size):
        async for coro_batch_result in pool.iterrun(coro_batch, yield_when=aio.FIRST_COMPLETED):
            yield coro_batch_result
