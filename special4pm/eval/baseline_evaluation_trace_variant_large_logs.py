import math
import math
import numpy as np
import pandas as pd
import pm4py
from matplotlib import pyplot as plt
from special4pm.visualization import visualization

from build.src.special4pm.species import species_retrieval
from special4pm.estimation import SpeciesEstimator
from special4pm.simulation.simulation import simulate_model
from functools import partial
from tqdm import tqdm
from special4pm.species import retrieve_species_n_gram


def init_estimator(step_size):
    estimator = SpeciesEstimator(step_size=step_size)
    estimator.register("3-gram", partial(retrieve_species_n_gram, n=3))
    estimator.register("tv", species_retrieval.retrieve_species_trace_variant)
    return estimator


def evaluate_model(path, name, repetitions, log_size, true_values):
    net, im, fm = pm4py.read_pnml(path)
    estimations = []

    for _ in tqdm(range(repetitions), "Simulating Model "):
        log = simulate_model(net, im, fm, log_size)
        step_size = int(len(log) / 200)
        estimator = init_estimator(step_size=step_size)
        estimator.apply(log, verbose=False)
        estimations.append(estimator)

    df = pd.concat([x.to_dataFrame() for x in estimations])

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
        visualization.plot_rank_abundance(estimations[0],species, False, save_to="fig/"+name + "_" + species + "_rank_abundance.pdf")

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
        plt.savefig("fig/"+name + "_" + species     + "_effort.pdf", format="pdf")
        #plt.show()
        plt.close()


true_values_net_3 ={
    "1-gram_abundance_estimate_d0":9,
    "1-gram_abundance_estimate_d1":9,
    "1-gram_abundance_estimate_d2":9,
    "1-gram_incidence_estimate_d0":9,
    "1-gram_incidence_estimate_d1":9,
    "1-gram_incidence_estimate_d2":9,
    "2-gram_abundance_estimate_d0":81,
    "2-gram_abundance_estimate_d1":81,
    "2-gram_abundance_estimate_d2":81,
    "2-gram_incidence_estimate_d0":81,
    "2-gram_incidence_estimate_d1":81,
    "2-gram_incidence_estimate_d2":81,
    "3-gram_abundance_estimate_d0":729,
    "3-gram_abundance_estimate_d1":729,
    "3-gram_abundance_estimate_d2":729,
    "3-gram_incidence_estimate_d0":729,
    "3-gram_incidence_estimate_d1":729,
    "3-gram_incidence_estimate_d2":729,
    "4-gram_abundance_estimate_d0":6561,
    "4-gram_abundance_estimate_d1":6561,
    "4-gram_abundance_estimate_d2":6561,
    "4-gram_incidence_estimate_d0":6561,
    "4-gram_incidence_estimate_d1":6561,
    "4-gram_incidence_estimate_d2":6551,
    "5-gram_abundance_estimate_d0":59049,
    "5-gram_abundance_estimate_d1":59049,
    "5-gram_abundance_estimate_d2":59049,
    "5-gram_incidence_estimate_d0":59049,
    "5-gram_incidence_estimate_d1":59049,
    "5-gram_incidence_estimate_d2":59049,
    "tv_abundance_estimate_d0": math.inf,
    "tv_abundance_estimate_d1": math.inf,
    "tv_abundance_estimate_d2": math.inf,
    "tv_incidence_estimate_d0": math.inf,
    "tv_incidence_estimate_d1": math.inf,
    "tv_incidence_estimate_d2": math.inf
}

true_values_net_7 ={
    "1-gram_abundance_estimate_d0":9,
    "1-gram_abundance_estimate_d1":7.70,
    "1-gram_abundance_estimate_d2":6.63,
    "1-gram_incidence_estimate_d0":9,
    "1-gram_incidence_estimate_d1":8.36,
    "1-gram_incidence_estimate_d2":7.81,
    "2-gram_abundance_estimate_d0":18,
    "2-gram_abundance_estimate_d1":15.66,
    "2-gram_abundance_estimate_d2":12.90,
    "2-gram_incidence_estimate_d0":18,
    "2-gram_incidence_estimate_d1":15.86,
    "2-gram_incidence_estimate_d2":14.39,
    "3-gram_abundance_estimate_d0":33,
    "3-gram_abundance_estimate_d1":26.92,
    "3-gram_abundance_estimate_d2":22.85,
    "3-gram_incidence_estimate_d0":33,
    "3-gram_incidence_estimate_d1":27.91,
    "3-gram_incidence_estimate_d2":24.73,
    "4-gram_abundance_estimate_d0":55,
    "4-gram_abundance_estimate_d1":43.10,
    "4-gram_abundance_estimate_d2":34.84,
    "4-gram_incidence_estimate_d0":55,
    "4-gram_incidence_estimate_d1":44.10,
    "4-gram_incidence_estimate_d2":36.42,
    "5-gram_abundance_estimate_d0":85,
    "5-gram_abundance_estimate_d1":66.72,
    "5-gram_abundance_estimate_d2":54.35,
    "5-gram_incidence_estimate_d0":85,
    "5-gram_incidence_estimate_d1":67.90,
    "5-gram_incidence_estimate_d2":56.40,
    "tv_abundance_estimate_d0": math.inf,
    "tv_abundance_estimate_d1": math.inf,
    "tv_abundance_estimate_d2": math.inf,
    "tv_incidence_estimate_d0": math.inf,
    "tv_incidence_estimate_d1": math.inf,
    "tv_incidence_estimate_d2": math.inf
}

true_values_net_8 ={
    "1-gram_abundance_estimate_d0":9,
    "1-gram_abundance_estimate_d1":9,
    "1-gram_abundance_estimate_d2":9,
    "1-gram_incidence_estimate_d0":9,
    "1-gram_incidence_estimate_d1":9,
    "1-gram_incidence_estimate_d2":9,
    "2-gram_abundance_estimate_d0":72,
    "2-gram_abundance_estimate_d1":72,
    "2-gram_abundance_estimate_d2":72,
    "2-gram_incidence_estimate_d0":72,
    "2-gram_incidence_estimate_d1":72,
    "2-gram_incidence_estimate_d2":72,
    "3-gram_abundance_estimate_d0":504,
    "3-gram_abundance_estimate_d1":504,
    "3-gram_abundance_estimate_d2":504,
    "3-gram_incidence_estimate_d0":504,
    "3-gram_incidence_estimate_d1":504,
    "3-gram_incidence_estimate_d2":504,
    "4-gram_abundance_estimate_d0":3024,
    "4-gram_abundance_estimate_d1":3024,
    "4-gram_abundance_estimate_d2":3024,
    "4-gram_incidence_estimate_d0":3024,
    "4-gram_incidence_estimate_d1":3024,
    "4-gram_incidence_estimate_d2":3024,
    "5-gram_abundance_estimate_d0":15120,
    "5-gram_abundance_estimate_d1":15120,
    "5-gram_abundance_estimate_d2":15120,
    "5-gram_incidence_estimate_d0":15120,
    "5-gram_incidence_estimate_d1":15120,
    "5-gram_incidence_estimate_d2":15120,
    "tv_abundance_estimate_d0": 362880,
    "tv_abundance_estimate_d1": 362880,
    "tv_abundance_estimate_d2": 362880,
    "tv_incidence_estimate_d0": 362880,
    "tv_incidence_estimate_d1": 362880,
    "tv_incidence_estimate_d2": 362880
}


#Used in Evaluation
evaluate_model("../../nets/net_3.pnml", "baseline_eval_model_3_EXTRA_LARGE_TEST", 200, 20000, true_values_net_3)
evaluate_model("../../nets/net_7.pnml", "baseline_eval_model_7_EXTRA_LARGE_TEST", 200, 20000, true_values_net_7)
evaluate_model("../../nets/net_8.pnml", "baseline_eval_model_8_EXTRA_LARGE_TEST", 200, 20000, true_values_net_8)