import copy
import math

import numpy as np
import pandas as pd
from functools import partial

from matplotlib import pyplot as plt
import pm4py
import random

from special4pm.simulation.simulation import simulate_model
from special4pm.visualization import visualization

from special4pm.estimation.species_estimator import SpeciesEstimator
from tqdm import tqdm

from special4pm.species import retrieve_species_n_gram, species_retrieval


def init_estimator(step_size):
    estimator = SpeciesEstimator(step_size=step_size)
    estimator.register("1-gram", partial(retrieve_species_n_gram, n=1))
    estimator.register("2-gram", partial(retrieve_species_n_gram, n=2))
    estimator.register("3-gram", partial(retrieve_species_n_gram, n=3))
    estimator.register("4-gram", partial(retrieve_species_n_gram, n=4))
    estimator.register("5-gram", partial(retrieve_species_n_gram, n=5))
    estimator.register("tv", species_retrieval.retrieve_species_trace_variant)
    return estimator


def profile_logs(logs):
    step_size = int(len(logs[0]) / 200)
    e = []
    for log in tqdm(logs, desc='Evaluating logs'):
        estimator = init_estimator(step_size=step_size)
        estimator.apply(log, verbose=False)
        e.append(estimator)
    return e


def evaluate_model(logs, name, true_values):
    estimations = profile_logs(logs)

    df = pd.concat([x.to_dataFrame() for x in estimations])
    df.to_csv("out/" + name + ".csv", index=False)

    df_last_values_only = pd.concat([x.to_dataFrame(include_all=False) for x in estimations])
    df_last_values_only.to_csv("out/" + name + "_final_only.csv", index=False)

    stats = df.groupby(["species", "metric", "observation"])["value"].agg(
        ['count', 'mean', 'var', 'std', 'sem']).reset_index()
    ci95_hi = []
    ci95_lo = []
    true = []
    bias = []
    rmse = []

    for i in stats.index:
        species, metric, _, c, m, v, s, sem = stats.loc[i]
        ci95_hi.append(m + 1.96 * s)
        ci95_lo.append(m - 1.96 * s)
        if species+"_"+metric in true_values:
            bias_row = true_values[species+"_"+metric] - m
            bias.append(bias_row)
            rmse.append(math.sqrt(bias_row ** 2 + v))
            true.append(true_values[species+"_"+metric])
        else:
            bias.append(-1)
            rmse.append(-1)
            true.append(-1)
    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo
    stats['bias'] = bias
    stats['rmse'] = rmse
    stats['true_values'] = true

    stats.to_csv("out/" + name + "_stats.csv", index=False)

    for species in estimations[0].metrics.keys():
        print("Evaluating " + name + ", " + species)
        obs_ids = df[(df["species"] == species) & (df["metric"] == "incidence_no_observations")]["value"].to_list()
        no_obs = len(
            stats[(stats["species"] == species) & (stats["metric"] == "incidence_no_observations")]["mean"].to_list())

        ### Rank Abundance Curves
        visualization.plot_rank_abundance(estimations[0], species, False,
                                          "fig/" + name + "_" + species + "_rank_abundance.pdf")

        plt.rcParams['figure.figsize'] = [3 * 9, 5]
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20

        ### Diversity Profiles
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey='all')
        #print(stats[(stats["species"] == species) & (stats["metric"] == "incidence_estimate_d0")])
        ax1.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_estimate_d0")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_estimate_d0")]["ci95_hi"],
                         alpha=0.5)
        ax1.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_estimate_d0")]["mean"],
                 label="Estimated")

        ax1.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d0")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d0")]["ci95_hi"],
                         alpha=0.5)
        ax1.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d0")]["mean"],
                 label="Observed")
        ax1.set_xticks([0, no_obs], [0, int(obs_ids[-1])])

        ax1.legend(fontsize=20)
        ax1.set_title("q=0", fontsize=28)
        ax1.set_xlabel("Sample Size", fontsize=24)
        ax1.set_ylabel("Hill number", fontsize=24)

        ax2.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_estimate_d1")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_estimate_d1")]["ci95_hi"],
                         alpha=0.5)
        ax2.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_estimate_d1")]["mean"],
                 label="Estimated")

        ax2.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d1")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d1")]["ci95_hi"],
                         alpha=0.5)
        ax2.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d1")]["mean"],
                 label="Observed")
        ax2.set_xticks([0, no_obs], [0, int(obs_ids[-1])])

        ax2.legend(fontsize=20)
        ax2.set_title("q=1", fontsize=28)
        ax2.set_xlabel("Sample Size", fontsize=24)

        ax3.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_estimate_d2")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_estimate_d2")]["ci95_hi"],
                         alpha=0.5)
        ax3.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_estimate_d2")]["mean"],
                 label="Estimated")

        ax3.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d2")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d2")]["ci95_hi"],
                         alpha=0.5)
        ax3.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d2")]["mean"],
                 label="Observed")
        ax3.set_xticks([0, no_obs], [0, int(obs_ids[-1])])

        ax3.legend(fontsize=20)
        ax3.set_title("q=2", fontsize=28)
        ax3.set_xlabel("Sample Size", fontsize=24)

        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig("fig/" + name + "_" + species + "_diversity_profile.pdf", format="pdf")
        #plt.show()
        plt.close()

        plt.rcParams['figure.figsize'] = [9, 5]
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20

        ### Completeness Profile
        plt.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_c0")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_c0")]["ci95_hi"],
                         alpha=0.5)
        plt.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_c0")]["mean"],
                 label="Completness")

        plt.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_c1")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_c1")]["ci95_hi"],
                         alpha=0.5)
        plt.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_c1")]["mean"], label="Coverage")
        plt.xticks([0, no_obs], [0, int(obs_ids[-1])])

        plt.legend(fontsize=20)
        plt.xlabel("Sample Size", fontsize=24)

        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig("fig/" + name + "_" + species + "_completeness_profile.pdf", format="pdf")
        #plt.show()
        plt.close()

        plt.rcParams['figure.figsize'] = [9, 5]
        plt.rcParams['xtick.labelsize'] = 20
        plt.rcParams['ytick.labelsize'] = 20

        ### Sampling Effort
        plt.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_l_0.99")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_l_0.99")]["ci95_hi"],
                         alpha=0.5)
        plt.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_l_0.99")]["mean"], label="l=.99")

        plt.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_l_0.95")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_l_0.95")]["ci95_hi"],
                         alpha=0.5)
        plt.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_l_0.95")]["mean"], label="l=.95")

        plt.fill_between(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_l_0.9")]["ci95_lo"],
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_l_0.9")]["ci95_hi"],
                         alpha=0.5)
        plt.plot(np.linspace(0, no_obs, no_obs),
                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_l_0.9")]["mean"], label="l=.90")
        plt.xticks([0, no_obs], [0, int(obs_ids[-1])])

        plt.legend(fontsize=20)
        plt.xlabel("Sample Size", fontsize=24)

        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig("fig/" + name + "_" + species + "_effort.pdf", format="pdf")
        plt.close()

def compare_rank_abundances(base, small, medium, large, name):
    l = [base, small, medium, large]
    e_b = SpeciesEstimator(step_size=None)
    e_b.register("2-gram", partial(retrieve_species_n_gram, n=2))
    e_b.apply(base)
    e_s = SpeciesEstimator(step_size=None)
    e_s.register("2-gram", partial(retrieve_species_n_gram, n=2))
    e_s.apply(small)
    e_m = SpeciesEstimator(step_size=None)
    e_m.register("2-gram", partial(retrieve_species_n_gram, n=2))
    e_m.apply(medium)
    e_l = SpeciesEstimator(step_size=None)
    e_l.register("2-gram", partial(retrieve_species_n_gram, n=2))
    e_l.apply(large)
    labels = ("Original","0.01% Noise","0.1% Noise","1% Noise")

    plt.rcParams['figure.figsize'] = [6 * 4, 3.5]
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey="row", layout="constrained")
    for x,e,l in zip([ax1, ax2, ax3, ax4],(e_b.metrics["2-gram"].reference_sample_abundance, e_s.metrics["2-gram"].reference_sample_abundance, e_m.metrics["2-gram"].reference_sample_abundance, e_l.metrics["2-gram"].reference_sample_abundance),labels):
        reference_sample = e
        species_count = sum(reference_sample.values())
        reference_values_sorted = sorted(list(reference_sample.values()), reverse=True)
        no_species = len(reference_sample)

        x.fill_between(np.linspace(0, no_species, no_species, endpoint=False),
                         reference_values_sorted,
                         [0 for _ in range(no_species)], alpha=0.4)
        x.plot(reference_values_sorted)
        x.set_xlabel("Species Rank", fontsize=24)
        x.set_ylabel("Occurrences", fontsize=24)

        x.set_xticks([0, no_species - 1], [1, no_species])
        x.set_yticks([0, max(reference_values_sorted)],
                   [0, max(reference_values_sorted)])
        x.set_title(l, fontsize = 28)
    plt.tight_layout()
    plt.savefig("fig/"+name+"_ABSOLUTE", format="pdf")
    plt.close()

    f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey="row", layout="constrained")
    for x,e,l in zip((ax1, ax2, ax3, ax4),(e_b.metrics["2-gram"].reference_sample_incidence, e_s.metrics["2-gram"].reference_sample_incidence, e_m.metrics["2-gram"].reference_sample_incidence, e_l.metrics["2-gram"].reference_sample_incidence),labels):
        reference_sample = e
        species_count = sum(reference_sample.values())
        reference_values_sorted = [x/species_count for x in sorted(list(reference_sample.values()))]
        no_species = len(reference_sample)

        x.fill_between(np.linspace(0, no_species, no_species, endpoint=False),
                         reference_values_sorted,
                         [0 for _ in range(no_species)], alpha=0.4)
        x.plot(reference_values_sorted)
        x.set_xlabel("Species Rank", fontsize=24)
        x.set_ylabel("Occurrences", fontsize=24)
        x.set_xticks([0, no_species - 1], [1, no_species])
        x.set_yticks([0, max(reference_values_sorted)],
                   [0, max(reference_values_sorted)])
        plt.yticks([0, max(reference_values_sorted)],[0, '%.2f'%(max(reference_values_sorted))])
        x.set_title(l, fontsize = 28)

    plt.tight_layout()
    plt.savefig("fig/"+name+"_RELATIVE", format="pdf")
    plt.close()


true_values_small = {
    "1-gram_abundance_estimate_d0": 5,
    "1-gram_abundance_estimate_d1": 5,
    "1-gram_abundance_estimate_d2": 5,
    "1-gram_incidence_estimate_d0": 5,
    "1-gram_incidence_estimate_d1": 5,
    "1-gram_incidence_estimate_d2": 5,
    "2-gram_abundance_estimate_d0": 36,
    "2-gram_abundance_estimate_d1": 36,
    "2-gram_abundance_estimate_d2": 36,
    "2-gram_incidence_estimate_d0": 36,
    "2-gram_incidence_estimate_d1": 36,
    "2-gram_incidence_estimate_d2": 36,
    "3-gram_abundance_estimate_d0": 176,
    "3-gram_abundance_estimate_d1": 176,
    "3-gram_abundance_estimate_d2": 176,
    "3-gram_incidence_estimate_d0": 176,
    "3-gram_incidence_estimate_d1": 176,
    "3-gram_incidence_estimate_d2": 176,
    "4-gram_abundance_estimate_d0": 876,
    "4-gram_abundance_estimate_d1": 876,
    "4-gram_abundance_estimate_d2": 876,
    "4-gram_incidence_estimate_d0": 876,
    "4-gram_incidence_estimate_d1": 876,
    "4-gram_incidence_estimate_d2": 876,
    "5-gram_abundance_estimate_d0": 4376,
    "5-gram_abundance_estimate_d1": 4376,
    "5-gram_abundance_estimate_d2": 4376,
    "5-gram_incidence_estimate_d0": 4376,
    "5-gram_incidence_estimate_d1": 4376,
    "5-gram_incidence_estimate_d2": 4376,
    "tv_abundance_estimate_d0": math.inf,
    "tv_abundance_estimate_d1": math.inf,
    "tv_abundance_estimate_d2": math.inf,
    "tv_incidence_estimate_d0": math.inf,
    "tv_incidence_estimate_d1": math.inf,
    "tv_incidence_estimate_d2": math.inf
}

true_values_medium = {
    "1-gram_abundance_estimate_d0": 5,
    "1-gram_abundance_estimate_d1": 5,
    "1-gram_abundance_estimate_d2": 5,
    "1-gram_incidence_estimate_d0": 5,
    "1-gram_incidence_estimate_d1": 5,
    "1-gram_incidence_estimate_d2": 5,
    "2-gram_abundance_estimate_d0": 36,
    "2-gram_abundance_estimate_d1": 36,
    "2-gram_abundance_estimate_d2": 36,
    "2-gram_incidence_estimate_d0": 36,
    "2-gram_incidence_estimate_d1": 36,
    "2-gram_incidence_estimate_d2": 36,
    "3-gram_abundance_estimate_d0": 176,
    "3-gram_abundance_estimate_d1": 176,
    "3-gram_abundance_estimate_d2": 176,
    "3-gram_incidence_estimate_d0": 176,
    "3-gram_incidence_estimate_d1": 176,
    "3-gram_incidence_estimate_d2": 176,
    "4-gram_abundance_estimate_d0": 876,
    "4-gram_abundance_estimate_d1": 876,
    "4-gram_abundance_estimate_d2": 876,
    "4-gram_incidence_estimate_d0": 876,
    "4-gram_incidence_estimate_d1": 876,
    "4-gram_incidence_estimate_d2": 876,
    "5-gram_abundance_estimate_d0": 4376,
    "5-gram_abundance_estimate_d1": 4376,
    "5-gram_abundance_estimate_d2": 4376,
    "5-gram_incidence_estimate_d0": 4376,
    "5-gram_incidence_estimate_d1": 4376,
    "5-gram_incidence_estimate_d2": 4376,
    "tv_abundance_estimate_d0": math.inf,
    "tv_abundance_estimate_d1": math.inf,
    "tv_abundance_estimate_d2": math.inf,
    "tv_incidence_estimate_d0": math.inf,
    "tv_incidence_estimate_d1": math.inf,
    "tv_incidence_estimate_d2": math.inf
}

true_values_large = {
    "1-gram_abundance_estimate_d0":4,
    "1-gram_abundance_estimate_d1":3.59,
    "1-gram_abundance_estimate_d2":3.33,
    "1-gram_incidence_estimate_d0":4,
    "1-gram_incidence_estimate_d1":3.94,
    "1-gram_incidence_estimate_d2":3.89,
    "2-gram_abundance_estimate_d0":16,
    "2-gram_abundance_estimate_d1":12.92,
    "2-gram_abundance_estimate_d2":11.10,
    "2-gram_incidence_estimate_d0":16,
    "2-gram_incidence_estimate_d1":14.50,
    "2-gram_incidence_estimate_d2":13.51,
    "3-gram_abundance_estimate_d0":64,
    "3-gram_abundance_estimate_d1":46.44,
    "3-gram_abundance_estimate_d2":36.98,
    "3-gram_incidence_estimate_d0":64,
    "3-gram_incidence_estimate_d1":50.43,
    "3-gram_incidence_estimate_d2":42.90,
    "4-gram_abundance_estimate_d0":256,
    "4-gram_abundance_estimate_d1":166.88,
    "4-gram_abundance_estimate_d2":123.06,
    "4-gram_incidence_estimate_d0":256,
    "4-gram_incidence_estimate_d1":174.38,
    "4-gram_incidence_estimate_d2":134.28,
    "5-gram_abundance_estimate_d0":1024,
    "5-gram_abundance_estimate_d1":599.39,
    "5-gram_abundance_estimate_d2":409.26,
    "5-gram_incidence_estimate_d0":1024,
    "5-gram_incidence_estimate_d1":611.67,
    "5-gram_incidence_estimate_d2":428.61,
    "tv_abundance_estimate_d0": math.inf,
    "tv_abundance_estimate_d1": math.inf,
    "tv_abundance_estimate_d2": math.inf,
    "tv_incidence_estimate_d0": math.inf,
    "tv_incidence_estimate_d1": math.inf,
    "tv_incidence_estimate_d2": math.inf
}

net, im, fm = pm4py.read_pnml("../../nets/base_net.pnml")

simulated_logs = [simulate_model(net, im, fm, 1000) for _ in tqdm(range(1), "Simulating Logs")]

events = []
for log in simulated_logs:
    log_events = []
    for i, tr in enumerate(log):
        for j, ev in enumerate(tr):
            log_events.append((i, j))
    events.append(log_events)

noisy_logs_small = []
#log consisting of 0.01% noise
for (log, log_events) in zip(simulated_logs, events):
    log_copy = copy.deepcopy(log)
    evs = random.sample(log_events, math.ceil(len(log_events) / 10000))
    for ev in evs:
        log_copy[ev[0]][ev[1]]["concept:name"] = str(random.getrandbits(128))
    noisy_logs_small.append(log_copy)

evaluate_model(noisy_logs_small, "noise_eval_small", true_values_small)

noisy_logs_medium = []
#log consisting of 0.1% noise
for (log, log_events) in zip(simulated_logs, events):
    log_copy = copy.deepcopy(log)
    evs = random.sample(log_events, math.ceil(len(log_events) / 1000))
    for ev in evs:
        log_copy[ev[0]][ev[1]]["concept:name"] = str(random.getrandbits(128))
    noisy_logs_medium.append(log_copy)

evaluate_model(noisy_logs_medium, "noise_eval_medium", true_values_medium)

noisy_logs_large = []
for (log, log_events) in zip(simulated_logs, events):
    #log consisting of 1% noise
    log_copy = copy.deepcopy(log)
    evs = random.sample(log_events, math.ceil(len(log_events) / 100))
    for ev in evs:
        log_copy[ev[0]][ev[1]]["concept:name"] = str(random.getrandbits(128))
    noisy_logs_large.append(log_copy)

evaluate_model(noisy_logs_large, "noise_eval_large", true_values_large)

compare_rank_abundances(simulated_logs[0],noisy_logs_small[0],noisy_logs_medium[0],noisy_logs_large[0], "noise_eval_rank_comparison")