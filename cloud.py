SEE_DISTANCE = 5
bucket_name = "experiments"

def check_stats(bucket_name, experiment, generation_num, agent_num):
    ff = f"experiment-{experiment}/generation-{generation_num}/agent-{agent_num}-stats.json"
    return ff in bucketstore.S3Bucket(bucket_name)

async def get_stats(obj):
    func_url = "http://gateway.openfaas.svc.cluster.local:8080/function/snake-stats"
    async with httpx.AsyncClient() as session:
        result_stat_code = 0
        while not check_stats(**obj):
            if result_stat_code != 200:
                result_stat_code = (await session.post(func_url, json=obj, timeout=None)).status_code
            await asyncio.sleep(1)
    # print(f"Stats for agent {obj['agent_num']} have been written")

def map_stats(bucket_name, experiment, generation_num, total_num_agents):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    ## set random kv's
    loop.run_until_complete(
        exaust(
            pool_run(
                loop,
                (
                 get_stats({
                     "bucket_name": bucket_name,
                     "experiment": experiment,
                     "generation_num": generation_num,
                     "agent_num": agent_num
                 })
                 for agent_num in range(total_num_agents)
                ),
                size=100, batch_size=100
            )
        ),
    )

def check_mutations(bucket_name, experiment, generation_num_read, generation_num_write,
                    agent_num_read, agent_num_write, num_mutations):
    ff = f"experiment-{experiment}/generation-{generation_num_write}/agent-{agent_num_write}.json"
    result = ff in bucketstore.S3Bucket(bucket_name)
    return result

async def get_mutations(obj, loop=None):
    func_url = "http://gateway.openfaas.svc.cluster.local:8080/function/snake-mutate"
    async with httpx.AsyncClient() as session:
        result_stat_code = 0
        while not check_mutations(**obj):
            if result_stat_code != 200:
                result_stat_code = (await session.post(func_url, json=obj, timeout=None)).status_code
            await asyncio.sleep(1)
    # print(f"Mutations for agent {obj['agent_num_read']} have been written to {obj['agent_num_write']}")
    return obj['agent_num_write']

def map_mutations(bucket_name, experiment, generation_num, from_to_mut_select_agents):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(
        exaust(
            pool_run(
                loop,
                (
                 get_mutations(dict(
                    bucket_name=bucket_name,
                    experiment=experiment,
                    generation_num_read=str(generation_num),
                    generation_num_write=str(generation_num+1),
                    agent_num_read=str(agent_num_read),
                    agent_num_write=str(agent_num_write),
                    num_mutations=str(num_mutations)
                ))
                for agent_num_read, agent_num_write, num_mutations
                in from_to_mut_select_agents
                ),
                size=64, batch_size=32
            )
        ),
    )

async def get_mutate_stats(bucket_name, experiment_name, latest_generation_num,
                           next_generation_num, agent_num_read, agent_num_write,
                           num_mutations):
    await get_mutations(dict(
        bucket_name=bucket_name,
        experiment=experiment_name,
        generation_num_read=str(latest_generation_num),
        generation_num_write=str(next_generation_num),
        agent_num_read=str(agent_num_read),
        agent_num_write=str(agent_num_write),
        num_mutations=str(num_mutations)
    ))
    await get_stats({
        "bucket_name": bucket_name,
        "experiment": experiment_name,
        "generation_num": str(next_generation_num),
        "agent_num": str(agent_num_write)
    })

def map_mutate_stats(bucket_name, experiment_name, latest_generation_num,
                     next_generation_num, from_to_mutations_agents):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(
        tqdm.asyncio.tqdm.gather(*[
            get_mutate_stats(
                bucket_name=bucket_name,
                experiment_name=experiment_name,
                latest_generation_num=str(latest_generation_num),
                next_generation_num=str(next_generation_num),
                agent_num_read=str(agent_num_read),
                agent_num_write=str(agent_num_write),
                num_mutations=str(num_mutations)
            )
            for agent_num_read, agent_num_write, num_mutations
            in from_to_mutations_agents
        ], ascii=True),
    )

    # loop.run_until_complete(
    #     exaust(
    #         pool_run(
    #             loop,
    #             (
    #              get_mutate_stats(
    #                  bucket_name=bucket_name,
    #                  experiment_name=experiment_name,
    #                  latest_generation_num=str(latest_generation_num),
    #                  next_generation_num=str(next_generation_num),
    #                  agent_num_read=str(agent_num_read),
    #                  agent_num_write=str(agent_num_write),
    #                  num_mutations=str(num_mutations)
    #             )
    #             for agent_num_read, agent_num_write, num_mutations
    #             in from_to_mutations_agents
    #             ),
    #             size=1024, batch_size=512
    #         )
    #     ),
    # )

def cut_selection(bucket_name, experiment, generation_num, perc):
    b = bucketstore.S3Bucket(bucket_name)
    agents_fitnesses = []
    for agent_stats_filename in filter(lambda f:"stats" in f,b.list(prefix=f"experiment-{experiment_name}/generation-{generation_num}")):
        agent_num = ''.join(filter(lambda elem:elem.isdigit(),agent_stats_filename.split('/')[-1]))
        fitness = json.loads(b[agent_stats_filename].decode())['fitness']
        agents_fitnesses.append((
            agent_stats_filename,
            int(agent_num),
            fitness
        ))

    sorted_agents = sorted(agents_fitnesses, key=lambda i:i[2], reverse=True)
    best_agents_index = floor(len(sorted_agents)*perc)
    best_agents = list(sorted_agents[:best_agents_index])
    return best_agents

def tournement_selection(bucket_name, experiment, generation_num, perc):
    b = bucketstore.S3Bucket(bucket_name)
    agents_fitnesses = []
    for agent_stats_filename in filter(lambda f:"stats" in f,b.list(prefix=f"experiment-{experiment_name}/generation-{generation_num}")):
        agent_num = ''.join(filter(lambda elem:elem.isdigit(),agent_stats_filename.split('/')[-1]))
        fitness = json.loads(b[agent_stats_filename].decode())['fitness']
        agents_fitnesses.append((
            agent_stats_filename,
            int(agent_num),
            fitness
        ))

    num_selected_agents = floor(len(agents_fitnesses)*perc)
    selected_agents = []
    partition_size, remainder = floor(len(agents_fitnesses) / num_selected_agents), len(agents_fitnesses) % num_selected_agents
    for p in range(num_selected_agents):
        tournament,agents_fitnesses = rand_n_from_list(partition_size, agents_fitnesses)
        selected_agents.append(sorted(tournament, key=lambda i:i[2], reverse=True)[0])

    return selected_agents

## expected in sorted order
def mush_sample_agents(samples, agent_nums, mush_factor):
    weighting = mushed_weighting(len(agent_nums), mush_factor)
    sampled_agents = []
    for _ in range(samples):
        sampled_agent = np.random.choice(agent_nums, p=weighting)
        sampled_agents.append(sampled_agent)
    return sampled_agents

def seed_generation(bucket, experiment, generation_num, population_num):
    print(f"Seeding generation {generation_num} ...")
    for agent_num in range(population_num):
        new_agent = Agent(SEE_DISTANCE,None).set_basic()
        write_key = f"experiment-{experiment}/generation-{generation_num}/agent-{agent_num}.json"
        bucket[write_key] = serialize(new_agent)
        print(f"Written agent num: {agent_num}")

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def iter_keys(bucket_name, prefix):
    bucket = bucketstore.S3Bucket(bucket_name)
    paginator = bucket._boto_s3.meta.client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]

population = 200
bucket = bucketstore.S3Bucket(bucket_name, create=True)
experiment_name = "big-boi-plays"
def run_generation():
    counts = Counter(int(elem.split('/')[1].split('-')[1]) for elem in filter(lambda f:"stats" in f,iter_keys(bucket_name, prefix=f"experiment-{experiment_name}")))

    if counts == Counter():
        print("Running generation 0")
        seed_generation(bucket, experiment_name, 0, population)
        map_stats(bucket_name, experiment_name, 0, population)
        latest_generation_num = 0
        return latest_generation_num
    else:
        _, latest_generation_num = max((count, name) for name, count in counts.items())
        latest_generation_num = int(latest_generation_num)
        print(f"Running generation {latest_generation_num}")

        # selected_agents = cut_selection(bucket_name, experiment_name,
        #                                 generation_num=latest_generation_num, perc=0.1)
        selected_agents = tournement_selection(
            bucket_name, experiment_name,generation_num=latest_generation_num, perc=0.1
        )
        selected_agents = sorted(selected_agents, key=lambda i:i[1], reverse=True)

        print("Best Agent:")
        print(bucket[selected_agents[0][0]])
        selected_agent_nums = list(map(lambda i:i[1], selected_agents))
        sampled_agent_nums = mush_sample_agents(
            population-len(selected_agents),
            selected_agent_nums,
            0.50,
        )

        from_to_mutations_agents = [
            (from_a, to_a, 0) if to_a<len(selected_agent_nums) else (from_a, to_a, 100)
            for (to_a, from_a) in enumerate(
                chain(
                    selected_agent_nums,
                    sampled_agent_nums
                )
            )
        ]

        map_mutate_stats(
            bucket_name, experiment_name,
            latest_generation_num=int(latest_generation_num),
            next_generation_num=int(latest_generation_num)+1,
            from_to_mutations_agents=from_to_mutations_agents
        )

        return int(latest_generation_num)+1

def read_input():
    return json.loads("".join(sys.stdin.read()))

def run_store_stats(bucket_name, experiment, generation_num, agent_num):

    ## setup the read and write
    bucket = bucketstore.S3Bucket(bucket_name)
    read_key = f"experiment-{experiment}/generation-{generation_num}/agent-{agent_num}.json"
    write_key = f"experiment-{experiment}/generation-{generation_num}/agent-{agent_num}-stats.json"

    ## Exit early if the computatino has already been done
    if write_key in bucket:
        return

    ## get the agent
    agent = deserialize(bucket[read_key].decode(), env=get_game())

    ## write the stats
    bucket[write_key] = json.dumps(agent.stats())

def run_store_mutations(bucket_name, experiment, generation_num_read, generation_num_write,
                        agent_num_read, agent_num_write, num_mutations):
    ## setup the read and write
    bucket = bucketstore.S3Bucket(bucket_name)
    read_key = f"experiment-{experiment}/generation-{generation_num_read}/agent-{agent_num_read}.json"
    write_key = f"experiment-{experiment}/generation-{generation_num_write}/agent-{agent_num_write}.json"

    ## Exit early if the computatino has already been done
    if write_key in bucket:
        return

    ## get the agent
    agent = deserialize(bucket[read_key].decode(), env=get_game())

    ## write the latest mutant
    # *_, lastest_agent = agent.mutate_n(num_mutations)
    for i in range(int(num_mutations)):
        agent.mutate()
    bucket[write_key] = serialize(agent)

if __name__ == "__main__":
    ## Get the input
    inp = read_input()

    ## Run the func
    if os.environ["FUNC_MODE"] == "mutate":
        run_store_mutations(**inp)
    elif os.environ["FUNC_MODE"] == "stats":
        run_store_stats(**inp)
    else:
        raise Exception("Bad mode")
