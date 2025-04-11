import math
from functools import partial

import numpy as np
import pandas as pd
from special4pm.visualization import visualization

import special4pm
from matplotlib import pyplot as plt

import pm4py
import copy

from tqdm import tqdm

from special4pm.estimation import SpeciesEstimator
from special4pm.simulation.simulation import simulate_model
from special4pm.species import retrieve_species_n_gram, species_retrieval

STEP_SIZE = 10

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


def evaluate_model(logs, name, true_values, drift=False):
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
        visualization.plot_rank_abundance(estimations[0],species, False, "fig/"+name + "_" + species + "_rank_abundance.pdf")

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

        if drift:
            ax1.hlines(true_values[species+"_incidence_estimate_d0"], xmin=no_obs/2, xmax=no_obs, color = "grey", ls="--")
        else:
            ax1.axhline(true_values[species+"_incidence_estimate_d0"], color = "grey", ls="--")

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

        if drift:
            ax2.hlines(true_values[species+"_incidence_estimate_d1"], xmin=no_obs/2, xmax=no_obs, color = "grey", ls="--")
        else:
            ax2.axhline(true_values[species+"_incidence_estimate_d1"], color = "grey", ls="--")

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

        if drift:
            ax3.hlines(true_values[species+"_incidence_estimate_d2"], xmin=no_obs/2, xmax=no_obs, color = "grey", ls="--")
        else:
            ax3.axhline(true_values[species+"_incidence_estimate_d2"], color = "grey", ls="--")

        ax3.set_xticks([0, no_obs], [0, int(obs_ids[-1])])

        ax3.legend(fontsize=20)
        ax3.set_title("q=2", fontsize=28)
        ax3.set_xlabel("Sample Size", fontsize=24)

        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig("fig/"+name + "_" + species + "_diversity_profile.pdf", format="pdf")
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
        plt.savefig("fig/"+name + "_" + species + "_completeness_profile.pdf", format="pdf")
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
        plt.savefig("fig/"+name + "_" + species + "_effort.pdf", format="pdf")
        plt.close()

true_values_base_net ={
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

true_values_added_e ={
    "1-gram_abundance_estimate_d0":5,
    "1-gram_abundance_estimate_d1":4.50,
    "1-gram_abundance_estimate_d2":4.17,
    "1-gram_incidence_estimate_d0":5,
    "1-gram_incidence_estimate_d1":4.92,
    "1-gram_incidence_estimate_d2":4.85,
    "2-gram_abundance_estimate_d0":25,
    "2-gram_abundance_estimate_d1":20.29,
    "2-gram_abundance_estimate_d2":17.37,
    "2-gram_incidence_estimate_d0":25,
    "2-gram_incidence_estimate_d1":22.28,
    "2-gram_incidence_estimate_d2":20.41,
    "3-gram_abundance_estimate_d0":125,
    "3-gram_abundance_estimate_d1":91.36,
    "3-gram_abundance_estimate_d2":72.40,
    "3-gram_incidence_estimate_d0":125,
    "3-gram_incidence_estimate_d1":96.14,
    "3-gram_incidence_estimate_d2":79.34,
    "4-gram_abundance_estimate_d0":625,
    "4-gram_abundance_estimate_d1":411.45,
    "4-gram_abundance_estimate_d2":301.80,
    "4-gram_incidence_estimate_d0":625,
    "4-gram_incidence_estimate_d1":419.85,
    "4-gram_incidence_estimate_d2":313.90,
    "5-gram_abundance_estimate_d0":3125,
    "5-gram_abundance_estimate_d1":1852.54,
    "5-gram_abundance_estimate_d2":1257.81,
    "5-gram_incidence_estimate_d0":3125,
    "5-gram_incidence_estimate_d1":1865.58,
    "5-gram_incidence_estimate_d2":1276.94,
    "tv_abundance_estimate_d0": math.inf,
    "tv_abundance_estimate_d1": math.inf,
    "tv_abundance_estimate_d2": math.inf,
    "tv_incidence_estimate_d0": math.inf,
    "tv_incidence_estimate_d1": math.inf,
    "tv_incidence_estimate_d2": math.inf
}

true_values_removed_b ={
    "1-gram_abundance_estimate_d0":3,
    "1-gram_abundance_estimate_d1":2.27,
    "1-gram_abundance_estimate_d2":1.85,
    "1-gram_incidence_estimate_d0":3,
    "1-gram_incidence_estimate_d1":2.92,
    "1-gram_incidence_estimate_d2":2.86,
    "2-gram_abundance_estimate_d0":9,
    "2-gram_abundance_estimate_d1":4.97,
    "2-gram_abundance_estimate_d2":3.43,
    "2-gram_incidence_estimate_d0":9,
    "2-gram_incidence_estimate_d1":7.51,
    "2-gram_incidence_estimate_d2":6.70,
    "3-gram_abundance_estimate_d0":27,
    "3-gram_abundance_estimate_d1":11.09,
    "3-gram_abundance_estimate_d2":6.36,
    "3-gram_incidence_estimate_d0":27,
    "3-gram_incidence_estimate_d1":17.38,
    "3-gram_incidence_estimate_d2":13.71,
    "4-gram_abundance_estimate_d0":81,
    "4-gram_abundance_estimate_d1":24.74,
    "4-gram_abundance_estimate_d2":11.79,
    "4-gram_incidence_estimate_d0":81,
    "4-gram_incidence_estimate_d1":37.51,
    "4-gram_incidence_estimate_d2":25.39,
    "5-gram_abundance_estimate_d0":243,
    "5-gram_abundance_estimate_d1":55.15,
    "5-gram_abundance_estimate_d2":21.83,
    "5-gram_incidence_estimate_d0":243,
    "5-gram_incidence_estimate_d1":78.70,
    "5-gram_incidence_estimate_d2":45.01,
    "tv_abundance_estimate_d0": math.inf,
    "tv_abundance_estimate_d1": math.inf,
    "tv_abundance_estimate_d2": math.inf,
    "tv_incidence_estimate_d0": math.inf,
    "tv_incidence_estimate_d1": math.inf,
    "tv_incidence_estimate_d2": math.inf
}

true_values_equal ={
    "1-gram_abundance_estimate_d0":5,
    "1-gram_abundance_estimate_d1":5,
    "1-gram_abundance_estimate_d2":5,
    "1-gram_incidence_estimate_d0":5,
    "1-gram_incidence_estimate_d1":5,
    "1-gram_incidence_estimate_d2":5,
    "2-gram_abundance_estimate_d0":25,
    "2-gram_abundance_estimate_d1":25,
    "2-gram_abundance_estimate_d2":25,
    "2-gram_incidence_estimate_d0":25,
    "2-gram_incidence_estimate_d1":25,
    "2-gram_incidence_estimate_d2":25,
    "3-gram_abundance_estimate_d0":125,
    "3-gram_abundance_estimate_d1":125,
    "3-gram_abundance_estimate_d2":125,
    "3-gram_incidence_estimate_d0":125,
    "3-gram_incidence_estimate_d1":125,
    "3-gram_incidence_estimate_d2":125,
    "4-gram_abundance_estimate_d0":625,
    "4-gram_abundance_estimate_d1":625,
    "4-gram_abundance_estimate_d2":625,
    "4-gram_incidence_estimate_d0":625,
    "4-gram_incidence_estimate_d1":625,
    "4-gram_incidence_estimate_d2":625,
    "5-gram_abundance_estimate_d0":3125,
    "5-gram_abundance_estimate_d1":3125,
    "5-gram_abundance_estimate_d2":3125,
    "5-gram_incidence_estimate_d0":3125,
    "5-gram_incidence_estimate_d1":3125,
    "5-gram_incidence_estimate_d2":3125,
    "tv_abundance_estimate_d0": math.inf,
    "tv_abundance_estimate_d1": math.inf,
    "tv_abundance_estimate_d2": math.inf,
    "tv_incidence_estimate_d0": math.inf,
    "tv_incidence_estimate_d1": math.inf,
    "tv_incidence_estimate_d2": math.inf
}


net, im, fm = pm4py.read_pnml("../../nets/base_net.pnml")
#pm4py.view_petri_net(net, im, fm)

base_logs = [simulate_model(net, im, fm, 1000) for _ in tqdm(range(200), "Simulating Base Net")]

net, im, fm = pm4py.read_pnml("../../nets/net_added_e.pnml")
#pm4py.view_petri_net(net, im, fm)

added_e_logs = [simulate_model(net, im, fm, 1000) for _ in tqdm(range(200), "Simulating Added Net")]

net, im, fm = pm4py.read_pnml("../../nets/net_removed_b.pnml")
#pm4py.view_petri_net(net, im, fm)

removed_b_logs = [simulate_model(net, im, fm, 1000) for _ in tqdm(range(200), "Simulating Removed Net")]

net, im, fm = pm4py.read_pnml("../../nets/net_equal_prob.pnml")
#pm4py.view_petri_net(net, im, fm)

equal_prob_logs = [simulate_model(net, im, fm, 1000) for _ in tqdm(range(200), "Simulating Equal Net")]

evaluate_model(base_logs, "drift_eval_base_model", true_values_base_net)
evaluate_model(added_e_logs, "drift_eval_added_activity", true_values_added_e)
evaluate_model(removed_b_logs, "drift_eval_removed_activity", true_values_removed_b)
evaluate_model(equal_prob_logs, "drift_eval_equal_probability", true_values_equal)

extended_logs = []
for (base_log,extend_log) in zip(base_logs,added_e_logs):
    a = copy.deepcopy(base_log)
    b = copy.deepcopy(extend_log)
    for x in b:
        a.append(x)
    extended_logs.append(a)
evaluate_model(extended_logs, "drift_eval_base_into_added_LARGE", true_values_added_e, drift=True)

extended_logs = []
for (base_log,extend_log) in zip(base_logs,removed_b_logs):
    a = copy.deepcopy(base_log)
    b = copy.deepcopy(extend_log)
    for x in b:
        a.append(x)
    extended_logs.append(a)
evaluate_model(extended_logs, "drift_eval_base_into_removed_LARGE", true_values_removed_b, drift=True)

extended_logs = []
for (base_log,extend_log) in zip(base_logs,equal_prob_logs):
    a = copy.deepcopy(base_log)
    b = copy.deepcopy(extend_log)
    for x in b:
        a.append(x)
    extended_logs.append(a)
evaluate_model(extended_logs, "drift_eval_base_into_equal_LARGE", true_values_equal, drift=True)
