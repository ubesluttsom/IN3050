from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib as mlp
import itertools as it
import numpy as np
import pandas as pd
import seaborn as sns
import timeit

# Initialize Seaborn
sns.set_theme()

# Initialize randomness generator
random = np.random.default_rng()

# Lists of city coordinates
city_coords={
    "Barcelona":        [2.154007, 41.390205],
    "Belgrade":         [20.46,    44.79],
    "Berlin":           [13.40,    52.52],
    "Brussels":         [4.35,     50.85],
    "Bucharest":        [26.10,    44.44],
    "Budapest":         [19.04,    47.50],
    "Copenhagen":       [12.57,    55.68],
    "Dublin":           [-6.27,    53.35],
    "Hamburg":          [9.99,     53.55],
    "Istanbul":         [28.98,    41.02],
    "Kiev":             [30.52,    50.45],
    "London":           [-0.12,    51.51],
    "Madrid":           [-3.70,    40.42],
    "Milan":            [9.19,     45.46],
    "Moscow":           [37.62,    55.75],
    "Munich":           [11.58,    48.14],
    "Paris":            [2.35,     48.86],
    "Prague":           [14.42,    50.07],
    "Rome":             [12.50,    41.90],
    "Saint Petersburg": [30.31,    59.94],
    "Sofia":            [23.32,    42.70],
    "Stockholm":        [18.06,    60.33],
    "Vienna":           [16.36,    48.21],
    "Warsaw":           [21.02,    52.24]
  }

# Convert to list format, and calculate values for map size.
city_coord_list = [[x, y] for x, y in city_coords.values()]
center = np.mean(np.array(city_coord_list)[:], axis=0)
min_lon, min_lat = np.min(np.array(city_coord_list)[:], axis=0)
max_lon, max_lat = np.max(np.array(city_coord_list)[:], axis=0)

# Import distances from CSV files.
import csv
with open("european_cities.csv", "r") as f:
    distances = np.array(list(csv.reader(f, delimiter=';')))
    cities = distances[0]
 
fig, ax = plt.subplots(figsize=(10,10))

def plot_plan(city_order, axis=None, text_markers=False):
    """ Plotting plans using Basemap library.
    `city_order` must be a sequence of integers.
    """

    if not axis:
        fig, ax = plt.subplots(figsize=(10,10))
    else:
        ax = axis

    # I'm using Basemap here to get the nice curved lines between coordinates,
    # representing the actual distance on a sphere. I`m still using the
    # Mercator projection, though, since the coordinate arithmetic is simpler.
    # The map is automatically initialized with the full tour centered.
    europe = Basemap(
            llcrnrlon=min_lon-5,
            llcrnrlat=min_lat-5,
            urcrnrlon=max_lon+5,
            urcrnrlat=max_lat+5.,
            lon_0=center[0],
            lat_0=center[1],
            resolution='c',
            projection='merc',
            ax=ax,
    )
    europe.fillcontinents('green', alpha=0.2)

    # Plot lines between all cities in circuit.
    lon1, lat1 = city_coord_list[city_order[-1]]
    for (lon2, lat2) in [city_coord_list[i] for i in city_order]:
        europe.drawgreatcircle(lon1, lat1, lon2, lat2, color='C3') 
        europe.plot(lon1, lat1, 'oC3', latlon=True)
        lon1, lat1 = lon2, lat2

    if axis:
        return ax
    else:
        plt.savefig("plan.pdf", format="pdf")
        return city_order

def city_names(cycle:np.ndarray):
    """ Convert a cycle/permutation of city indexes to a list of names.
    """
    # Tile the header of the data array `len(cycle)` times, creating an ndarray
    # with the header repeated in “rows” downward. Then for each row `take` the
    # indices specified by each `cycle`.
    return np.take(np.tile(distances[0][:len(cycle)], [len(cycle), 1]), (cycle))


# ┌───────────────────┐
# │ EXHAUSTIVE SEARCH │
# └───────────────────┘

def exhaustive_search(n=6):
    """ Test all orderings of the first `n` cities.
    """
    best_fitness = np.inf
    best_cyle    = np.arange(n)
    for cycle in cycles(range(n)):
        if (new_fitness := fitness(cycle)) < best_fitness:
            best_fitness = new_fitness
            best_cycle   = cycle
    return best_cycle

def cycles(items):
    """ Generates next cycles with a single fixed point.
    """
    # This saves us $n! - (n-1)!$ iterations. See:
    # [https://en.wikipedia.org/wiki/Stirling_numbers_of_the_first_kind].
    # Think of it as always starting in Barcelona; keep the fist city fixed.
    first = (items[0],)
    for cycle in it.permutations(items[1:]):
        yield np.array(first + cycle)

def fitness(perm, distances=distances):
    """ Evaluate the fitness of a city ordering (in integer form)
    """
    # Some awful data slicing here, sorry. Extracts the relative distances,
    # using the permutation numbers as indexes. Then sum them up.
    return np.sum(distances[:,(perm)][(perm+1),:][(perm),(perm-1)], dtype=float)


# ┌───────────────┐
# │ HILL CLIMBING │
# └───────────────┘

def hill_climbing(order=None, n=6, n_swaps=10):
    """ Hill climbing algorithm, more or less copied from the Marsland book.
    Starts with a random ordering of `n` cities, and does `n_swaps` randomly,
    only accepting those yielding fitness improvements.
    """

    if not type(order) is np.ndarray:
        order = random.permutation(n)
    best_fitness = fitness(order)

    # Do `n_swaps` iterations.
    for index in range(n_swaps):
        # Generate two random integers.
        city1, city2 = random.integers(n, size=2)
        if city1 != city2:
            # `np.select` works like this: IF any any of the conditions in the
            # first list, THEN select the respective element from the second
            # list. Selects the first, if multiple True conditions. I.e, here,
            # swap the two randomly selected cities.
            new_order = np.select([order == city1, order == city2, True],
                                  [city2,          city1,          order])
            # Evaluate new fitness, keep swap if it's an improvement.
            if (new_fitness := fitness(new_order)) < best_fitness:
                    order = new_order
                    best_fitness = new_fitness
    return order


# ┌───────────────────┐
# │ GENETIC ALGORITHM │
# └───────────────────┘

def genetic_algorithm(n, size=58, generations=100, λ=58, μ=10, s=1.5,
                      lamarckian=None, plot=False):
    """ See @[eiben-smith, p. 26]

    The `lamarckian` argument signals a hybrid algorithm, and takes
    an integer specifying the number of swaps performed on each solution.
    """
    population = generate_population(n, size=size)
    highest_fitness = fitness(population[0])
    termination_counter = 0

    if plot:
        mean, best, worst = np.full([3, generations], np.nan)

    for g in range(generations):

        if plot:
            pf = population_fitness(population)
            mean[g], best[g], worst[g] = np.mean(pf), np.min(pf), np.max(pf)

        parents    = parent_selection(population, λ, μ, s)

        if lamarckian:
            parents = lamarckian_selection(parents, lamarckian)

        offspring  = recombine(parents)
        offspring  = mutate(offspring)
        population = survivor_selection(parents, offspring, size=size)

        # Setting a termination condition. If no improvement after 15
        # generations, break.
        if (contender := fitness(population[0])) < highest_fitness:
            highest_fitness = contender
            termination_counter = 0
        else:
            termination_counter += 1
        if termination_counter >= 30:
            break

    if plot:
        genetic_algorithm_plot(
            pd.DataFrame({
                'mean': mean,
                'best': best,
                'worst': worst,
                'size': size,
                'generations': generations,
                'λ': λ,
                'μ': μ,
                's': s,
            }),
            population[0]
        )

    return population[0]

def parent_selection(population, λ, μ, s):
    """ Choose appropriate parents from a population.
    """
    # TODO: Explain meaning of parameters.
    return stochastic_universal_sampling(fitness_sort(population), λ, μ, s)

def recombine(parents):
    """ Recombine a population of parents pairwise.
    """
    n, seq_len = np.shape(parents)

    # Rounding parents down to nearest even, possibly discarding, the worst
    # parent (since `parents` is still is sorted). Then shuffle them randomly
    # before `reshape`ing them into pairs.
    n = n - (n % 2)
    random.shuffle(parents[:n])
    parents    = parents[:n].reshape(n//2, 2, seq_len)
    offspring  = np.empty_like(parents)

    for index, pair in enumerate(parents):
        offspring[index] = order_crossover_pair(*pair)
    return offspring.reshape(n, seq_len)

def mutate(offspring):
    return np.apply_along_axis(inversion_mutation, 1, offspring)

def survivor_selection(parents, offspring, size=10):
    population = np.concatenate([parents, offspring])
    return fitness_sort(population)[:size]

def generate_population(n, size=10):
    """ Generate a `size` number of random sequences of length `n`.
    """
    population = np.tile(np.arange(n), [size, 1])
    return random.permuted(population, axis=1)

def population_fitness(population):
    """ Evaluate fitness of all individuals in a population.
    """
    return np.apply_along_axis(fitness, 1, population)

def fitness_sort(population):
    """ Sort a population by their fitness, descending.
    """
    return population[np.argsort(population_fitness(population))]

def stochastic_universal_sampling(parents, λ, μ, s=1.5):
    """ See @[eiben-smith, p. 84]
    """
    # TODO: These variables don't change much; possible to cache?
    probs        = linear_ranking_probs(len(parents), s, μ)
    random_value = random.uniform(high=1/λ)
    cdf          = np.cumsum(probs)

    mating_pool  = list()
    rank = 0
    while len(mating_pool) < λ:
        while random_value <= cdf[rank]:
            random_value += 1/λ
            mating_pool.append(parents[rank])
        rank += 1
    return np.array(mating_pool)

def linear_ranking_probs(n, s, μ):
    """ See @[eiben-smith, p. 82]
    """
    # TODO: Explain!
    return (2-s)/μ + (2*np.arange(n-1, -1, -1)*(s-1))/(μ*(μ-1)) 

def order_crossover_pair(P1, P2, cx_len=None):
    """ See @[eiben-smith, p. 73]
    """
    if (seq_len := len(P1)) != len(P2):
        raise ValueError('Sequence lengths must be equal!')
    if not cx_len:
        cx_len = seq_len // 2

    cx_start = random.integers(0, seq_len - cx_len)
    cx_end   = cx_start + cx_len
    
    C1, C2 = np.full([2, seq_len], -1)
    C1[cx_start:cx_end] = P1[cx_start:cx_end]
    C2[cx_start:cx_end] = P2[cx_start:cx_end]

    for (parent, child) in zip([P1, P2], [C2, C1]):
        j = 0
        for i in range(seq_len):
            gene = parent[(cx_end + i) % seq_len]
            if gene not in child:
                child[(cx_end + j) % seq_len] = gene
                j += 1

    return C1, C2

def inversion_mutation(child):
    """ Mutate using inversion. See [@eiben-smith, p. 69]
    """
    cuts  = random.integers(1, len(child), size=2)
    child = np.roll(child, cuts[0])
    return np.concatenate((child[cuts[1]-1::-1], child[cuts[1]:]))


# ┌───────────────────┐
# │ HYBRID ALGORITHMS │
# └───────────────────┘

def lamarckian_selection(population, swaps=10):
    return np.apply_along_axis(hill_climbing, 1, population,
                               n_swaps=swaps, n=population.shape[1])

# ┌──────────┐
# │ PLOTTERS │
# └──────────┘

def genetic_algorithm_plot(data, best_plan):
    """ Helper for plotting genetic algorithm data.
    """
    fig, ax = plt.subplots(2, 1, figsize=(5,7))

    # Plot mean, worst, and best plan for all generations.
    sns.lineplot(
            data=data[['mean', 'worst', 'best']],
            ax=ax[0],
            drawstyle='steps-post',
    ) 
    ax[0].set_ylabel('Distance')
    ax[0].set_xlabel('Generation')

    plot_plan(best_plan, axis=ax[1])

    # Adjust layout and save figure to file.
    plt.tight_layout()
    fig.savefig("genetic_algorithm_plot.pdf", format="pdf")


def time_algorithms(
            runs=10,
            cities=24,
            start=2,
            size=58,
            generations=100,
            λ=58,
            μ=10,
            s=1.5,
            la_swaps=10,
            hc_swaps=1000,
            write=False,
            algorithms=['es', 'hc', 'ga', 'la'],
    ):
    """ Time algorithm runs.
    """

    ga_common_params = {
            "λ": λ,
            "μ": μ,
            "s": s,
            "generations": generations,
            "size": size,
        }

    print(ga_common_params | {'la_swaps': la_swaps, 'hc_swaps': hc_swaps, 'algorithms': algorithms })

    # DO RUNS USING `timeit`
    # ----------------------

    # Change the template of `timeit` module to also retrieve return values.
    # (Source code)[https://github.com/python/cpython/blob/3.10/Lib/timeit.py]
    timeit.template = \
    'def inner(_it, _timer{init}):       \n' \
    '    {setup}                         \n' \
    '    _t0 = _timer()                  \n' \
    '    for _i in _it:                  \n' \
    '        return_value = {stmt}       \n' \
    '    _t1 = _timer()                  \n' \
    '    return _t1 - _t0, return_value    '
    timeit_kwargs = {'number': 1, 'setup': f'params = {ga_common_params}','globals': globals()}

    new_data = list()

    run_es = True if 'es' in algorithms else False
    run_hc = True if 'hc' in algorithms else False
    run_ga = True if 'ga' in algorithms else False
    run_la = True if 'la' in algorithms else False

    for n in range(start, cities + 1):
        print(f"Timing runs for {n} cities ", end='', flush=True)
        
        if n > 10:
            run_es = False

        for run in range(runs):
            print(".", end='', flush=True)

            if run_es:
                es = timeit.timeit(f'exhaustive_search(n={n})', **timeit_kwargs)
                es = {"algorithm": 'es', "time": es[0], "plan": es[1]}
                new_data += [es]
            if run_hc:
                hc = timeit.timeit(f'hill_climbing(n={n}, n_swaps=1000)', **timeit_kwargs)
                hc = {"algorithm": 'hc', "time": hc[0], "plan": hc[1], "swaps": hc_swaps} | ga_common_params
                new_data += [hc]
            if run_ga:
                ga = timeit.timeit(f'genetic_algorithm(n={n}, **params)', **timeit_kwargs)
                ga = {"algorithm": 'ga', "time": ga[0], "plan": ga[1]} | ga_common_params
                new_data += [ga]
            if run_la:
                la = timeit.timeit(f'genetic_algorithm(n={n}, **params, lamarckian={la_swaps})', **timeit_kwargs)
                la = {"algorithm": 'la', "time": la[0], "plan": la[1], "swaps": la_swaps} | ga_common_params
                new_data += [la]

        print("")

    new_data = [pd.Series(dict) for dict in new_data]
    new_data = pd.concat(new_data, axis=1).T
    new_data['distance'] = new_data['plan'].apply(fitness)
    new_data['len'] = new_data['plan'].apply(len)

    if write:
        # I'm using JSON for storage; it's better than CSV for handling arrays.
        data = pd.concat([pd.read_json('runs.json'), new_data], ignore_index=True)
        data['plan'] = data['plan'].apply(np.array)
        data.to_json('runs.json')
    return new_data

def data_plotter(data: pd.DataFrame):
    """ Plot statistics from dataframe with runtimes.
    """

    plt.clf()
    if len(data['algorithm'].unique()) > 2:
        fig, axes = plt.subplots(3, 2, figsize=(10,15))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10,10))

    sns.lineplot(
            data=data,
            x="len",
            y="time",
            hue="algorithm",
            style="algorithm",
            ci=100,
            ax=axes[0,0]
    )

    sns.lineplot(
            data=data,
            x="len",
            y="distance",
            hue="algorithm",
            style="algorithm",
            ci=100,
            ax=axes[0,1]
    )

    # Plot best plans
    def best_plan(algorithm):
        return (data.sort_values(by=['len', 'distance'], ascending=[False, True])
                    .where(data.algorithm == algorithm)
                    .plan
                    .dropna()
                    .head(1)
                    .iloc[0])

    for alg, ax in zip(data['algorithm'].unique(), axes.flat[2:]):
        plot_plan(best_plan(alg), axis=ax)
        match alg:
            case 'es':
                ax.set_title(f'Exhaustive search, highest calculated')
            case 'hc':
                ax.set_title(f'Hill climbing')
            case 'ga':
                ax.set_title(f'Genetic algorithm')
            case 'la':
                ax.set_title(f'Hybrid algorithm, Lamarckian')

    # Adjust layout and save figure to file.
    plt.tight_layout()
    plt.savefig(f := "time.pdf", format="pdf")
    print(f'Saved output to {f}.')

def test_lots():
    i = 0
    while i < 10000:
        λ           = random.integers(2, 60)
        μ           = random.integers(2, 60)
        generations = random.integers(1, 100)
        swaps       = random.integers(1, 100)
        size        = random.integers(5, 100)
        s           = random.uniform(1.2, 1.8)
        if size >= μ:
            try:
                time_algorithms(
                    λ=λ,
                    μ=μ,
                    s=s,
                    generations=generations,
                    size=size,
                    la_swaps=swaps,
                    runs=1,
                    algorithms=['la', 'ga'],
                )
            except:
                pass
            i += 1

if __name__ == '__main__':

    # Generate figures and tables for report. You may run this to verify my
    # results.

    # EXHAUSTIVE SEARCH

    plt.clf()
    es = time_algorithms(cities=10, start=2, algorithms=['es'], runs=1)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    sns.lineplot(data=es, x='len', y='time',
                 hue='algorithm', style='algorithm', ax=axes[0,0])
    sns.lineplot(data=es, x='len', y='distance',
                 hue='algorithm', style='algorithm', ax=axes[0,1])
    axes[1,0].set_title('Exhaustive search, 6 cities')
    axes[1,1].set_title('Exhaustive search, 10 cities')
    plot_plan(es[es.len == 6]['plan'].iloc[0], axis=axes[1,0])
    plot_plan(es[es.len == 10]['plan'].iloc[0], axis=axes[1,1])
    plt.tight_layout()
    plt.savefig('es.pdf', format='pdf')
    es['cycles'] = es.len.apply(lambda x: np.math.factorial(x - 1))
    es

    # HILL CLIMBING

    plt.clf()
    hc = time_algorithms(cities=24, start=2, algorithms=['hc'], runs=20)
    eshc = pd.concat([es, hc])
    eshc = eshc.reset_index()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    sns.lineplot(data=eshc, x='len', y='time',
                 hue='algorithm', style='algorithm', ax=axes[0,0])
    sns.lineplot(data=eshc, x='len', y='distance',
                 hue='algorithm', style='algorithm', ax=axes[0,1])
    axes[1,0].set_title('Hill climbing, 10 cities')
    axes[1,1].set_title('Hill climbing, 24 cities')
    plot_plan(hc[hc.len == 10]['plan'].iloc[0], axis=axes[1,0])
    plot_plan(hc[hc.len == 24]['plan'].iloc[0], axis=axes[1,1])
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig('hc.pdf', format='pdf')
    hc[hc.len == 10].describe()
    hc[hc.len == 24].describe()

    # GENETIC ALGORITHM

    df = pd.read_json('runs.json')

    # Correlation matrix
    df[df.algorithm == 'ga'].corr().round(4)

    # Descriptive statistics. Fist on all data, then on 100 best solutions.
    df[df.algorithm == 'ga'][df.len == 24].describe().round(2)
    (df[df.algorithm == 'ga'][df.len == 24]
        .sort_values(by=['distance', 'time'])
        .head(100)
        .describe()
        .round(2))

    # Generate new data, and plot for different population sizes.
    def ga(size):
        ga = time_algorithms(cities=24, start=2, size=size, algorithms=['ga'], runs=20)
        ga['distance'] = ga.distance.apply(np.array)
        ga['time'] = ga.time.apply(np.array)
        print(f'# Genetic algorithm, population size {size}')
        print(ga[ga.len == 24].describe())
        return ga

    ga10  = ga(10)
    ga50  = ga(50)
    ga100 = ga(100)

    plt.clf()
    fig, axes = plt.subplots(2, 1, figsize=(5, 10))
    sns.lineplot(data=pd.concat([ga10, ga50, ga100]).reset_index(), ax=axes[0],
                 x='len', y='distance', hue='size', style='size', ci=100)
    sns.lineplot(data=pd.concat([ga10, ga50, ga100]).reset_index(), ax=axes[1],
                 x='len', y='time', hue='size', style='size', ci=100)
    plt.tight_layout()
    plt.savefig('ga-plot.pdf', format='pdf')

    plt.clf()
    fig, axes = plt.subplots(3, 1, figsize=(5, 15))
    plot_plan(ga10[ga10.len == 24].sort_values(by='distance')['plan'].iloc[0], axis=axes[0])
    plot_plan(ga50[ga50.len == 24].sort_values(by='distance')['plan'].iloc[0], axis=axes[1])
    plot_plan(ga100[ga100.len == 24].sort_values(by='distance')['plan'].iloc[0], axis=axes[2])
    axes[0].set_title('Genetic algorithm, population size 10')
    axes[1].set_title('Genetic algorithm, population size 50')
    axes[2].set_title('Genetic algorithm, population size 100')
    plt.tight_layout()
    plt.savefig('ga-plans.pdf', format='pdf')

    # Plot compared algorithms.
    eshcga = pd.concat([es,hc,ga50]).reset_index()
    plt.clf()
    fig, axes = plt.subplots(2, 1, figsize=(5,10))
    sns.lineplot(
            data=eshcga,
            x='len',
            y='time',
            hue='algorithm',
            style='algorithm',
            ci=100,
            ax=axes[0]
    )
    sns.lineplot(
            data=eshcga,
            x='len',
            y='distance',
            hue='algorithm',
            style='algorithm',
            ci=100,
            ax=axes[1]
    )
    plt.tight_layout()
    plt.savefig('algorithms.pdf', format='pdf')
  
# vim: ts=4 sw=4
