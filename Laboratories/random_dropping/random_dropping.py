import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
import argparse as ap


# %%
# fill n bins with random dropping policy
# input: n -> number of balls and bins
# output: max occupancy, min occupancy, avg occupancy
def random_dropping(n):
    bins = np.zeros(n, dtype=int)  # array of bins
    for i in range(n):
        bins[np.random.randint(0, n)] += 1  # select randomly 1 bin and add a ball

    max = bins.max()
    min = bins.min()
    mean = bins.mean()

    return max, min, mean


# %%
# fill n bins with random dropping policy
# input: n -> number of balls and bins, d -> number of bins chosen at each round
# output: max occupancy, min occupancy, avg occupancy
def random_dropping_load_balancing(n, d):
    bins = np.zeros(n, dtype=int)  # array of bins
    for i in range(0, n):
        selected_bins = np.random.randint(0, n, d)  # select at random d bins

        # find the bin from selected_bins that is least occupied
        min_value = n + 1
        min_index = -1
        for b in selected_bins:
            if bins[b] < min_value:
                min_value = bins[b]
                min_index = b

        bins[min_index] += 1  # increment the number of balls in the bin

    max = bins.max()
    min = bins.min()
    mean = bins.mean()

    return max, min, mean


# %%
# random dropping
def simulate_random_dropping():
    _d = dict()
    max_values_dict = dict()
    for n in np.logspace(2, 5, 50, endpoint=True, dtype=int):  # take numbers spaced on a log scale
        max, min, mean = random_dropping(n)
        # min usually is 0
        # avg always one
        th_value = math.log(n) / math.log(math.log(n))  # compute the theoretical value
        _d[n] = [max, th_value]
        max_values_dict[n] = max

        # create dataframe
        # df_rnd_dropping = pd.DataFrame.from_dict(_d, orient="index", columns=["max_occupancy", "th_max_occupancy"])
        # df_rnd_dropping.to_csv(f"rnd_dropping_simulation_{seed}.csv")

        # return a dataframe with the values of max occupancy for each n
    return pd.DataFrame.from_dict(max_values_dict, orient="index", columns=["max_occupancy"])


# %%
# load balancing d=2,4
def simulate_load_balancing(d):
    _d = dict()
    max_values_dict = dict()
    for n in np.logspace(2, 5, 50, endpoint=True, dtype=int):  # take numbers spaced on a log scale
        max, min, mean = random_dropping_load_balancing(n, d)
        # min usually is 0
        # avg always one
        th_value = math.log(math.log(n)) / math.log(d)  # compute the theoretical value
        _d[n] = [max, th_value]
        max_values_dict[n] = max

        # create dataframe df_load_balancing_2 = pd.DataFrame.from_dict(_d, orient="index", columns=["max_occupancy",
        # "th_max_occupancy"]) df_load_balancing_2.to_csv(f"load_balancing_{d}_{seed}.csv")

        # return a dataframe with the values of max occupancy for each n
    return pd.DataFrame.from_dict(max_values_dict, orient="index", columns=[f"max_occupancy_{d}"])


# %%
# input: number of runs
# iterate for each seed and simulates the dropping algorithms
# then compute and plot the confidence intervals

def run_simulation(n_runs):
    seeds = range(0, n_runs)
    dfs = []
    for seed in seeds:
        print(f"run with seed={seed}")
        np.random.seed(seed)
        print("\tsimulating random dropping...")
        df_rnd_dropping = simulate_random_dropping()  # simulate random dropping
        print("\tsimulating random dropping load balancing with d=2...")
        df_load_balancing_2 = simulate_load_balancing(2)  # simulate load balancing d=2
        print("\tsimulating random dropping load balancing with d=4...\n")
        df_load_balancing_4 = simulate_load_balancing(4)  # simulate load balancing d=4

        # create a dataframe for each seed
        dfs.append(pd.concat([df_rnd_dropping, df_load_balancing_2, df_load_balancing_4], axis=1))

    # compute mean and variance for each n between different runs
    mean_df = 0
    var_df = 0
    for df in dfs:
        mean_df += df

    mean_df /= len(dfs)

    for df in dfs:
        var_df = pow(df - mean_df, 2)

    var_df = 1 / (len(dfs) - 1) * var_df

    # 95% confidence interval
    t_crit = stats.t.ppf(q=0.95, df=len(dfs))
    down_i = mean_df - t_crit * np.sqrt(var_df) / np.sqrt(len(dfs))
    up_i = mean_df + t_crit * np.sqrt(var_df) / np.sqrt(len(dfs))

    # plot
    print("plot 95% confidence interval")
    plt.figure(figsize=(10, 7))

    plt.fill_between(x=dfs[0].index, y1=up_i["max_occupancy"], y2=down_i["max_occupancy"], alpha=.3)
    plt.fill_between(x=dfs[0].index, y1=up_i["max_occupancy_2"], y2=down_i["max_occupancy_2"], alpha=.3)
    plt.fill_between(x=dfs[0].index, y1=up_i["max_occupancy_4"], y2=down_i["max_occupancy_4"], alpha=.3)

    plt.plot(mean_df, marker="o", label=["random_dropping", "load_balancing d=2", "load_balancing d=4"])

    # plt.text(83, 7.15, "95% confidence interval")
    plt.legend()
    plt.grid()
    plt.xlabel("n")
    plt.ylabel("max_occupancy")
    plt.xscale("log")
    plt.show()


# %%
parser = ap.ArgumentParser()
parser.add_argument('--nRuns', type=int, default=10)
args = parser.parse_args()

# %%
run_simulation(args.nRuns)
