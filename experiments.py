from opteval import benchmark_func as bf
import sys
import matplotlib.pyplot as plt

from PPA import PPA


def collect_data(bench, NPOP, GENS, NMAX, OPT, runs):
    run_data = {"s":[], "e":[], "d":[]}
    for r in range(runs):
        s, e, d = PPA(bench, NPOP, GENS, NMAX, OPT)
        run_data['s'].append(s)
        run_data['e'].append(e)
        run_data['d'].append(d)
    return run_data

def get_right_gen(run_data, max_e):
    if not max_e is None:
        var = 'evot'
        total_evo_avg = [sum([run_data['d'][j][var][i]
                         for j in range(len(run_data['d']))]) / len(run_data['d'])
                    for i in range(len(run_data['d'][0][var]))]
        abs_l = [abs(i - max_e) for i in total_evo_avg]
        index = abs_l.index(min(abs_l))
    else:
        index = None

    return index

def average_data(run_data, max_e=None):
    # get average of the list
    avs = sum(run_data['s']) / len(run_data['s'])
    ave = sum(run_data['e']) / len(run_data['e'])

    index = get_right_gen(run_data, max_e)

    # get average of the data dictionary
    avd = {}
    for var in run_data['d'][0].keys():
        avd[var] = [sum([run_data['d'][j][var][i]
                   for j in range(len(run_data['d']))]) / len(run_data['d'])
                   for i in range(len(run_data['d'][0][var][:index]))]

    return avs, ave, avd

def plot_avg(avd, variables = None, extra_label='', benchname="Average over evaluations"):
    if variables is None:
        print("You can choose from: min, max and avg")

    for var in variables:
        plt.plot(avd['evot'], avd[var], label=var + extra_label)

    plt.legend()
    plt.ylabel("Value2")
    plt.xlabel("Evaluatations2")
    plt.title(benchname)

def box_scores(run_data, extra, OPT, N=0, max_e=1e5):
    # Get Score at evaluation = max_e
    index = get_right_gen(run_data, max_e)
    scores = [i[OPT][index] for i in run_data['d']]

    # Extra info about outliers ####
    cd = {}
    for sco in scores:
        sc = int(sco)
        if sc in cd:
            cd[sc] += 1
        else:
            cd[sc] = 1
    print(cd)
    ################################

    return scores

def name_check(name, ext, file_loc):
    i = 1
    while name+str(i)+ext in os.listdir(file_loc):
        i+=1

    return name+str(i)+ext

if __name__ == '__main__':
    # Experiment Vars
    runs = 100

    Benchmarks = {
                "Eggholder":bf.Eggholder(),
                "McCormick":bf.McCormick(),
                "Matyas":bf.Matyas(),
                "BukinN6":bf.BukinN6(),
                "ThreeHumpCamel":bf.ThreeHumpCamel(),
                "Easom":bf.Easom()
                }

    if len(sys.argv) < 2:
        print("usage: python script.py benchmark_name")
        sys.exit(1)
    elif not sys.argv[1] in Benchmarks.keys():
        print("please use:", str(list(Benchmarks.keys())))
        sys.exit(1)

    # Set variables for individual run
    NPOP: int = 50
    GENS: int = 100
    OPT: str = "min"
    benchname: str = sys.argv[1]
    max_eva: int = 10000

    # Make figure
    fig = plt.figure(figsize=(12,12))

    # loop though different experiment values
    exp = [4, 6, 8, 10]
    boxplotdata = []
    for n, e in enumerate(exp):
        print("experiment number:", n)
        # collect data from n runs.
        run_data = collect_data(Benchmarks[benchname], NPOP, GENS, e, OPT, runs)
        # calculte average for plotting
        avs, ave, avd = average_data(run_data, max_e=max_eva)

        # plot
        plt.subplot(121)
        plot_avg(avd, ['avg'], f'N={e}', benchname=f"Average value / evaluations: {benchname}")
        box = box_scores(run_data, f"N={e}", OPT=OPT,
                           max_e=max_eva, N=e)
        boxplotdata.append(box)

    plt.subplot(122)
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.labelright'] = True
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False
    plt.boxplot(boxplotdata)
    plt.title(f"Boxplot of end score at {max_eva} evaluations: {benchname}")
    plt.xlabel("number of offsprings")
    plt.ylabel("Value")
    # fig.savefig(name_check(f"analyse_{benchname}", ".png", None))
    plt.show()
