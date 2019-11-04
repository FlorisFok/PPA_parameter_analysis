"""
Plant propagation parameter tuning with different Benchmark fucntions by Floris Fok

Here we can test different parameter settings of the PPA in contrast to different Benchmarks.
The code is handwritten with exception of the benchmark function so small mistakes could be possible
in the execution of the PPA. Persesonal and non-commercial use only, Thx.

SOURCE FORMULA PPA:
Fireworks Algorithm versus Plant Propagation Algorithm by Wouter Vrielink and Daan van den Berg

SOURCE BENCHMARK FUNC:
https://github.com/keit0222/optimization-evaluation by @tomochiii and @keit0222
"""

from opteval import benchmark_func as bf
import random
import numpy


def fitness(x, y, bench):
    """
    Gets value of benchmark function for given input variable
    :param x: value1 (float)
    :param y: value2 (float)
    :param bench: the benchmark function (opteval.class)
    :return: output/fitness of function (flaot)
    """
    return bench.get_func_val([x, y])


def initalisation(npop, top, bot):
    """
    Random initialization of the first population/
    :param npop: size of population (int)
    :param top: upper bound (float)
    :param bot: lower bound (float)
    :return: list of new population (tuple of flaots)
    """
    pop = [(random.uniform(top[0], bot[0]), random.uniform(top[1], bot[1])) for i in range(npop)]
    return pop


def evaluate(pop, bench):
    """
    Gets values of benchmark function for given input variables
    :param pop: list of current population (tuple of floats)
    :param bench: the benchmark function (opteval.class)
    :return: list of scores (floats)
    """
    return [fitness(i[0], i[1], bench) for i in pop], len(pop)


def selectionMIN(pop, scores, NPOP):
    """
    Linear selection method to keep same population size (from low to high)
    :param pop: list of current population (tuples of floats)
    :param scores: list of current scores (float)
    :param NPOP: maximum size of population (int)
    :return: list of cropped population and list of cropped score.
    """
    both = list(zip(pop, scores))

    def sortit(x):
        return x[1]

    both.sort(key=sortit)
    both = both[:NPOP]
    return zip(*both)


def selectionMAX(pop, scores, NPOP):
    """
    Linear selection method to keep same population size (from high to low)
    :param pop: list of current population (tuples of floats)
    :param scores: list of current scores (float)
    :param NPOP: maximum size of population (int)
    :return: list of cropped population and list of cropped score.
    """
    both = list(zip(pop, scores))

    def sortit(x):
        return x[1]

    both.sort(key=sortit, reverse=True)
    both = both[:NPOP]
    return zip(*both)


def cal_fitness(ma, mi, value):
    """
    Calculate the fitness value relative to the others
    :param ma: maximum value in population (float)
    :param mi: minimum value in population (flaot)
    :param value: current value (float)
    :return: fitness (float)
    """
    if mi - ma == 0:
        ma += 1e-6
    f = ((ma - value) / (ma - mi))
    c_f = 0.5 * (numpy.tanh((4 * f) - 2) + 1)
    return c_f


def cal_offsprings(f, nmax):
    """
    Calculate the number of offsprings from the fitness
    :param f: fitness (flaot)
    :param nmax: max number of offsprings (int)
    :return: number of offsprings (int)
    """
    return int(numpy.ceil(random.random() * f * nmax))


def new_offspring(x, f, top, bot):
    """
    Creates a new offspring from the previous one
    :param x: previous value (float)
    :param f: fitness (float)
    :param top: upperbound (float)
    :param bot: lowerbound (float)
    :return: new value (float)
    """
    d = (1 - f) * random.uniform(-1, 1)
    x += (top - bot) * d

    # LIMIT correction
    if x > top:
        x = top
    elif x < bot:
        x = bot

    return x


def print_stats(scores, gen):
    """
    Prints the scores and generation number in a simple format
    :param scores: list of scores (floats)
    :param gen: generation (int)
    :return: None
    """
    print("Generation:", gen)
    print("Best =", round(scores[0], 2),
          "  Worst =", round(scores[-1], 2),
          "  Average = ", sum(scores) / NPOP)


def PPA(bench, NPOP, GENS, NMAX, OPT, plot=False, evaluation_max=1e100):
    """

    :param bench: The benchmark fuction which needs to be solved. (opteval.class)
    :param NPOP: Number of individuals in the population (int)
    :param GENS: maximum number of generations (int)
    :param NMAX: maximum number of offsprings (int)
    :param OPT: "min" or "max" imise the value
    :param plot: Bool, to display progress or not
    :return: Best score, number of evaluations and Data collected during the run (min, max, avg, evaluations)
    """
    # Get boundaries
    top = bench.max_search_range
    bot = bench.min_search_range
    total_evo = 0
    scores = numpy.zeros(NPOP)
    gen = 0

    # Initialize random
    new_pop = initalisation(NPOP, top, bot)
    pop = []
    scores = []

    data_dict = {
        'min': numpy.zeros(GENS),
        'max': numpy.zeros(GENS),
        'avg': numpy.zeros(GENS),
        'evot': numpy.zeros(GENS),
        'evod': numpy.zeros(GENS)
    }

    # Run for GENS times
    for gen in range(GENS):

        # Evaluate the populations
        new_scores, evaluations = evaluate(new_pop, bench)
        total_evo += evaluations

        # Merge populations
        pop = list(pop) + list(new_pop)
        scores = list(scores) + list(new_scores)

        if OPT == "min":
            pop, scores = selectionMIN(pop, scores, NPOP)
        else:
            pop, scores = selectionMAX(pop, scores, NPOP)

        # Generation variables
        ma = max(scores)
        mi = min(scores)
        new_pop = []

        # Data capture
        data_dict['min'][gen] = min(scores)
        data_dict['max'][gen] = max(scores)
        data_dict['avg'][gen] = sum(scores) / NPOP
        data_dict['evot'][gen] = total_evo
        data_dict['evod'][gen] = evaluations

        # Show progress
        if gen % int(GENS / 10) == 0 and plot:
            print_stats(scores, gen)

        # Calculate fitness and amount of offsprings for every individual
        for ind, value in zip(pop, scores):
            f = cal_fitness(ma, mi, value)
            n = cal_offsprings(f, NMAX)

            # Add N times a new offspring to the new population
            for new in range(n):
                new_pop.append((new_offspring(ind[0], f, top[0], bot[0]),
                                new_offspring(ind[1], f, top[1], bot[1])))


        # Early break
        if evaluation_max < total_evo:
            break

    # Final cut
    if OPT == "min":
        pop, scores = selectionMIN(pop, scores, NPOP)
    else:
        pop, scores = selectionMAX(pop, scores, NPOP)

    # Final plot
    if plot:
        print_stats(scores, GENS)

    # Relative score
    best = bench.global_optimum_solution

    # print("Offset", abs(scores[0] - best))
    return scores[0], total_evo, data_dict


if __name__ == '__main__':
    bench = bf.Eggholder()

    # Set variables
    NPOP = 10
    GENS = 50
    NMAX = 5
    OPT = "min"
    print("FINAL (value, evaluations):", PPA(bench, NPOP, GENS, NMAX, OPT, plot=True))
    print("GLOBAL:", bench.global_optimum_solution)
    bench.plot()
