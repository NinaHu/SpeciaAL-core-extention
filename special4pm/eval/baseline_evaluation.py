import math
import numpy as np
import pandas as pd
import pm4py
from matplotlib import pyplot as plt
from special4pm.visualization import visualization

#from build.src.special4pm.species import species_retrieval
from special4pm.species import species_retrieval
from special4pm.estimation import SpeciesEstimator
from special4pm.simulation.simulation import simulate_model
from functools import partial
from tqdm import tqdm
from special4pm.species import retrieve_species_n_gram
import os


def init_estimator(step_size):
    estimator = SpeciesEstimator(step_size=step_size, d1=False, d2=False, l_n=[])
    for n in range(1, 6):
        estimator.register(f"{n}-gram", partial(retrieve_species_n_gram, n=n))
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


def save_separate_csv_files(stats, name):
    output_folder = os.path.join("out", name)
    os.makedirs(output_folder, exist_ok=True)

    # save .csv data for every comb. of each n-gram and metric
    for (species, metric), group_data in stats.groupby(["species", "metric"]):
        filename = f"{species}_{metric}.csv"
        file_path = os.path.join(output_folder, filename)
        group_data.to_csv(file_path, index=False)

    # comb. all .csv data for every n-gram
    for n in range(1, 6):
        n_gram_name = f"{n}-gram"
        combined_data = stats[stats["species"] == n_gram_name]
        combined_file_path = os.path.join(output_folder, f"combined_{n_gram_name}_stats.csv")
        combined_data.to_csv(combined_file_path, index=False)

    print(f"Separate CSV files for individual metrics and combined n-grams saved in: {output_folder}")


def evaluate_model(path, name, repetitions, log_size, true_values):
    # load the model and simulate logs
    net, im, fm = pm4py.read_pnml(path)
    simulated_logs = []
    for _ in tqdm(range(repetitions), "Simulating Model "):
        simulated_logs.append(simulate_model(net, im, fm, log_size))
    estimations = profile_logs(simulated_logs)

    df = pd.concat([x.to_dataFrame() for x in estimations])
    df.to_csv("out/" + name + ".csv", index=False)

    df_last_values_only = pd.concat([x.to_dataFrame(include_all=False) for x in estimations])
    df_last_values_only.to_csv("out/" + name + "_final_only.csv", index=False)

    stats = df.groupby(["species", "metric", "observation"])["value"].agg(
        ['count', 'mean', 'var', 'std', 'sem']).reset_index()

    # DEBUG:
    print("First few rows of stats:")
    print(stats.head())

    ci95_hi = []
    ci95_lo = []
    true = []
    bias = []
    rmse = []

    # calculate statistics
    for i in stats.index:
        species, metric, _, c, m, v, s, sem = stats.loc[i]
        ci95_hi.append(m + 1.96 * s)
        ci95_lo.append(m - 1.96 * s)
        if species + "_" + metric in true_values:
            bias_row = true_values[species + "_" + metric] - m
            bias.append(bias_row)
            rmse.append(math.sqrt(bias_row ** 2 + v))
            true.append(true_values[species + "_" + metric])
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

    # save all .csv as separate files
    save_separate_csv_files(stats, name)

    stats.to_csv("out/" + name + "_stats.csv", index=False)
    # no_obs = len(rmse)
    ### Plotting: generate separate plot for each n-gram and for each metric
    for species in estimations[0].metrics.keys():
        print("Evaluating " + name + ", " + species)

        # DEBUG: check the length and the values for the dataframe
        obs_ids = df[(df["species"] == species) & (df["metric"] == "incidence_no_observations")]["value"].to_list()
        no_obs = len(
            stats[(stats["species"] == species) & (stats["metric"] == "incidence_no_observations")]["mean"].to_list())


        for metric in ["ace", "ace_modified", "chao1", "iChao1", "jackknife1_abundance", "jackknife1_incidence",
                       "jackknife2_abundance", "jackknife2_incidence", "ice", "ice_modified", "chao2", "iChao2"]:

            # for n in range (1,6)
            species_metric = species#f"{n}-gram"
            if species_metric in stats["species"].unique():  # check if metric exists in the statistics
                print(f"Evaluating {name}, {species_metric}")

                obs_ids = df[(df["species"] == species) & (df["metric"] == "incidence_no_observations")][
                    "value"].to_list()

                ### Rank Abundance Curves
                visualization.plot_rank_abundance(estimations[0], species, False,
                                                  "fig/" + name + "_" + species + "_" + metric + "_rank_abundance.pdf")

                plt.rcParams['figure.figsize'] = [3 * 9, 5]
                plt.rcParams['xtick.labelsize'] = 20
                plt.rcParams['ytick.labelsize'] = 20

                ### Diversity Profiles
                f, (ax1) = plt.subplots(nrows=1, ncols=1, sharey='all')
                no_obs = 201
                print(no_obs, len(stats[(stats["species"] == species) & (stats["metric"] == metric)]["ci95_lo"]))
                ax1.fill_between(np.linspace(0, no_obs, no_obs),
                                 stats[(stats["species"] == species) & (stats["metric"] == metric)]["ci95_lo"],
                                 stats[(stats["species"] == species) & (stats["metric"] == metric)]["ci95_hi"],
                                 alpha=0.5)
                ax1.plot(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == metric)]["mean"],
                         label="Estimated")

                ax1.fill_between(np.linspace(0, no_obs, no_obs),
                                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d0")][
                                     "ci95_lo"],
                                 stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d0")][
                                     "ci95_hi"],
                                 alpha=0.5)
                ax1.plot(np.linspace(0, no_obs, no_obs),
                         stats[(stats["species"] == species) & (stats["metric"] == "incidence_sample_d0")]["mean"],
                         label="Observed")

                # true value line
                ax1.axhline(true_values[species + "_incidence_estimate_d0"], color="grey", ls="--")

                ax1.set_xticks([0, no_obs], [0, int(obs_ids[-1])])
                ax1.legend(fontsize=20)
                ax1.set_title("q=0", fontsize=28)
                ax1.set_xlabel("Sample Size", fontsize=24)
                ax1.set_ylabel("Hill number", fontsize=24)

                plt.ylim(bottom=0)
                plt.tight_layout()
                plt.savefig("fig/" + name + "_" + species + "_" + metric + "_diversity_profile.pdf", format="pdf")

                # save the plot with metric name and n-gram in the filename
                plt.savefig(f"{metric}_{species}_plot.png", format="png")

                # plt.show()
                plt.close()

                plt.rcParams['figure.figsize'] = [9, 5]
                plt.rcParams['xtick.labelsize'] = 20
                plt.rcParams['ytick.labelsize'] = 20

        # combined plots for separate abundance-based and incidence-based metrics
        n_grams = range(1, 6)
        abundance_metrics = ["ace", "ace_modified", "chao1", "iChao1", "jackknife1_abundance", "jackknife2_abundance"]
        incidence_metrics = ["ice", "ice_modified", "chao2", "iChao2", "jackknife1_incidence", "jackknife2_incidence"]

        for n in n_grams:
            # plot size and axis labels
            plt.rcParams['figure.figsize'] = [20, 6]
            plt.rcParams['xtick.labelsize'] = 18
            plt.rcParams['ytick.labelsize'] = 18

            # plots for abundance-based metrics
            fig, ax1 = plt.subplots()
            species_metric = f"{n}-gram"

            for metric in abundance_metrics:
                if species_metric in stats["species"].unique():
                    ax1.fill_between(
                        np.linspace(0, log_size,
                                    len(stats[(stats["species"] == species_metric) & (stats["metric"] == metric)])),
                        stats[(stats["species"] == species_metric) & (stats["metric"] == metric)]["ci95_lo"],
                        stats[(stats["species"] == species_metric) & (stats["metric"] == metric)]["ci95_hi"],
                        alpha=0.2
                    )

                    # plot mean values for each metric
                    ax1.plot(
                        np.linspace(0, log_size,
                                    len(stats[(stats["species"] == species_metric) & (stats["metric"] == metric)])),
                        stats[(stats["species"] == species_metric) & (stats["metric"] == metric)]["mean"],
                        label=f"{metric}"
                    )

            # true value line
            if f"{species_metric}_abundance_estimate_d0" in true_values:
                ax1.axhline(true_values[f"{species_metric}_abundance_estimate_d0"], color="grey", ls="--",
                            label="True Value")

            # title and axis labels for abundance-based metrics
            ax1.set_title(f"Combined Diversity Profile for {n}-gram (Abundance-based)", fontsize=28)
            ax1.set_xlabel("Sample Size", fontsize=24)
            ax1.set_ylabel("Hill number", fontsize=24)
            ax1.legend(fontsize=14, loc='best', ncol=1, frameon=True, edgecolor='lightgray')
            plt.ylim(bottom=0)
            plt.tight_layout()

            # save the combined plot for abundance-based metrics
            plt.savefig(f"fig/combined_abundance_{n}-gram_diversity_profile.pdf", format="pdf", bbox_inches="tight")
            plt.savefig(f"fig/combined_abundance_{n}-gram_diversity_profile.png", format="png", bbox_inches="tight")
            plt.close()

            # plots for incidence-based metrics
            fig, ax2 = plt.subplots()

            for metric in incidence_metrics:
                if species_metric in stats["species"].unique():
                    ax2.fill_between(
                        np.linspace(0, log_size,
                                    len(stats[(stats["species"] == species_metric) & (stats["metric"] == metric)])),
                        stats[(stats["species"] == species_metric) & (stats["metric"] == metric)]["ci95_lo"],
                        stats[(stats["species"] == species_metric) & (stats["metric"] == metric)]["ci95_hi"],
                        alpha=0.2
                    )

                    # plot mean values for each metric
                    ax2.plot(
                        np.linspace(0, log_size,
                                    len(stats[(stats["species"] == species_metric) & (stats["metric"] == metric)])),
                        stats[(stats["species"] == species_metric) & (stats["metric"] == metric)]["mean"],
                        label=f"{metric}"
                    )

            # true value line
            if f"{species_metric}_incidence_estimate_d0" in true_values:
                ax2.axhline(true_values[f"{species_metric}_incidence_estimate_d0"], color="grey", ls="--",
                            label="True Value")

            # title and axis labels for incidence-based metrics
            ax2.set_title(f"Combined Diversity Profile for {n}-gram (Incidence-based)", fontsize=28)
            ax2.set_xlabel("Sample Size", fontsize=24)
            ax2.set_ylabel("Hill number", fontsize=24)
            ax2.legend(fontsize=14, loc='best', ncol=1, frameon=True, edgecolor='lightgray')
            plt.ylim(bottom=0)
            plt.tight_layout()

            # Save the plots
            plt.savefig(f"fig/combined_incidence_{n}-gram_diversity_profile.pdf", format="pdf", bbox_inches="tight")
            plt.savefig(f"fig/combined_incidence_{n}-gram_diversity_profile.png", format="png", bbox_inches="tight")
            plt.close()

            plt.rcParams['figure.figsize'] = [9, 5]
            plt.rcParams['xtick.labelsize'] = 20
            plt.rcParams['ytick.labelsize'] = 20

    # TRACE VARIANT PLOTS
        if "tv" in estimations[0].metrics.keys():
            print("Evaluating Trace Variants")

            # Plot size and axis labels
            plt.rcParams['figure.figsize'] = [26, 6]
            plt.rcParams['xtick.labelsize'] = 18
            plt.rcParams['ytick.labelsize'] = 18

            for n in range(1, 6):
                n_gram_name = f"{n}-gram"
                print(f"Creating TV vs {n_gram_name} plots")

                # Abundance-based Plot
                fig, ax1 = plt.subplots()

                for metric in abundance_metrics:
                    # layout for trace variants
                    tv_data = stats[(stats["species"] == "tv") & (stats["metric"] == metric)]
                    if not tv_data.empty:
                        ax1.fill_between(
                            np.linspace(0, log_size, len(tv_data)),
                            tv_data["ci95_lo"],
                            tv_data["ci95_hi"],
                            alpha=0.2
                        )
                        ax1.plot(
                            np.linspace(0, log_size, len(tv_data)),
                            tv_data["mean"],
                            label=f'TV {metric}',
                            linestyle='--'
                        )

                    # layout for n-grams
                    ngram_data = stats[(stats["species"] == n_gram_name) & (stats["metric"] == metric)]
                    if not ngram_data.empty:
                        ax1.fill_between(
                            np.linspace(0, log_size, len(ngram_data)),
                            ngram_data["ci95_lo"],
                            ngram_data["ci95_hi"],
                            alpha=0.2
                        )
                        ax1.plot(
                            np.linspace(0, log_size, len(ngram_data)),
                            ngram_data["mean"],
                            label=f'{n_gram_name} {metric}',
                            linestyle='-'
                        )

                # True value line
                if f"{n_gram_name}_abundance_estimate_d0" in true_values:
                    ax1.axhline(true_values[f"{n_gram_name}_abundance_estimate_d0"], color="grey", ls="--",
                                label="True Value")

                ax1.set_title(f"Combined Diversity Profile for {n}-gram vs Trace Variance (Abundance-based)", fontsize=28)
                ax1.set_xlabel("Sample Size", fontsize=24)
                ax1.set_ylabel("Hill number", fontsize=24)
                ax1.legend(fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, edgecolor='lightgray')
                plt.ylim(bottom=0)
                plt.tight_layout()

                plt.savefig(f"fig/combined_abundance_{n}-gram_vs_tv_diversity_profile.pdf", format="pdf",
                            bbox_inches="tight")
                plt.savefig(f"fig/combined_abundance_{n}-gram_vs_tv_diversity_profile.png", format="png",
                            bbox_inches="tight")
                plt.close()

                # Incidence-based Plot
                fig, ax2 = plt.subplots()

                for metric in incidence_metrics:
                    # layout for trace variants
                    tv_data = stats[(stats["species"] == "tv") & (stats["metric"] == metric)]
                    if not tv_data.empty:
                        ax2.fill_between(
                            np.linspace(0, log_size, len(tv_data)),
                            tv_data["ci95_lo"],
                            tv_data["ci95_hi"],
                            alpha=0.2
                        )
                        ax2.plot(
                            np.linspace(0, log_size, len(tv_data)),
                            tv_data["mean"],
                            label=f'TV {metric}',
                            linestyle='--'
                        )

                    # layout for n-grams
                    ngram_data = stats[(stats["species"] == n_gram_name) & (stats["metric"] == metric)]
                    if not ngram_data.empty:
                        ax2.fill_between(
                            np.linspace(0, log_size, len(ngram_data)),
                            ngram_data["ci95_lo"],
                            ngram_data["ci95_hi"],
                            alpha=0.2
                        )
                        ax2.plot(
                            np.linspace(0, log_size, len(ngram_data)),
                            ngram_data["mean"],
                            label=f'{n_gram_name} {metric}',
                            linestyle='-'
                        )

                # True value line
                if f"{n_gram_name}_incidence_estimate_d0" in true_values:
                    ax2.axhline(true_values[f"{n_gram_name}_incidence_estimate_d0"], color="grey", ls="--",
                                label="True Value")

                ax2.set_title(f"Combined Diversity Profile for {n}-gram vs Trace Variance (Incidence-based)", fontsize=28)
                ax2.set_xlabel("Sample Size", fontsize=24)
                ax2.set_ylabel("Hill number", fontsize=24)
                ax2.legend(fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, edgecolor='lightgray')
                plt.ylim(bottom=0)
                plt.tight_layout()

                plt.savefig(f"fig/combined_incidence_{n}-gram_vs_tv_diversity_profile.pdf", format="pdf",
                            bbox_inches="tight")
                plt.savefig(f"fig/combined_incidence_{n}-gram_vs_tv_diversity_profile.png", format="png",
                            bbox_inches="tight")
                plt.close()

            plt.rcParams['figure.figsize'] = [9, 5]
            plt.rcParams['xtick.labelsize'] = 20
            plt.rcParams['ytick.labelsize'] = 20


true_values_net_3 = {
    "1-gram_abundance_estimate_d0": 9,
    "1-gram_abundance_estimate_d1": 9,
    "1-gram_abundance_estimate_d2": 9,
    "1-gram_incidence_estimate_d0": 9,
    "1-gram_incidence_estimate_d1": 9,
    "1-gram_incidence_estimate_d2": 9,
    "2-gram_abundance_estimate_d0": 81,
    "2-gram_abundance_estimate_d1": 81,
    "2-gram_abundance_estimate_d2": 81,
    "2-gram_incidence_estimate_d0": 81,
    "2-gram_incidence_estimate_d1": 81,
    "2-gram_incidence_estimate_d2": 81,
    "3-gram_abundance_estimate_d0": 729,
    "3-gram_abundance_estimate_d1": 729,
    "3-gram_abundance_estimate_d2": 729,
    "3-gram_incidence_estimate_d0": 729,
    "3-gram_incidence_estimate_d1": 729,
    "3-gram_incidence_estimate_d2": 729,
    "4-gram_abundance_estimate_d0": 6561,
    "4-gram_abundance_estimate_d1": 6561,
    "4-gram_abundance_estimate_d2": 6561,
    "4-gram_incidence_estimate_d0": 6561,
    "4-gram_incidence_estimate_d1": 6561,
    "4-gram_incidence_estimate_d2": 6561,
    "5-gram_abundance_estimate_d0": 59049,
    "5-gram_abundance_estimate_d1": 59049,
    "5-gram_abundance_estimate_d2": 59049,
    "5-gram_incidence_estimate_d0": 59049,
    "5-gram_incidence_estimate_d1": 59049,
    "5-gram_incidence_estimate_d2": 59049,
    "tv_abundance_estimate_d0": math.inf,
    "tv_abundance_estimate_d1": math.inf,
    "tv_abundance_estimate_d2": math.inf,
    "tv_incidence_estimate_d0": math.inf,
    "tv_incidence_estimate_d1": math.inf,
    "tv_incidence_estimate_d2": math.inf,

    "1-gram_ace_abundance_estimate_d0": 9,
    "1-gram_ace_modified_abundance_estimate_d0": 9,
    "1-gram_chao1_abundance_estimate_d0": 9,
    "1-gram_iChao1_abundance_estimate_d0": 9,
    "1-gram_jackknife1_abundance_estimate_d0": 9,
    "1-gram_jackknife2_abundance_estimate_d0": 9,
    "1-gram_tv_abundance_based_d0": math.inf,

    "1-gram_ice_incidence_estimate_d0": 9,
    "1-gram_ice_modified_incidence_estimate_d0": 9,
    "1-gram_chao2_incidence_estimate_d0": 9,
    "1-gram_iChao2_incidence_estimate_d0": 9,
    "1-gram_jackknife1_incidence_estimate_d0": 9,
    "1-gram_jackknife2_incidence_estimate_d0": 9,
    "1-gram_tv_incidence_based_d0": math.inf,

    "2-gram_ace_abundance_estimate_d0": 81,
    "2-gram_ace_modified_abundance_estimate_d0": 81,
    "2-gram_chao1_abundance_estimate_d0": 81,
    "2-gram_iChao1_abundance_estimate_d0": 81,
    "2-gram_jackknife1_abundance_estimate_d0": 81,
    "2-gram_jackknife2_abundance_estimate_d0": 81,
    "2-gram_tv_abundance_based_d0": math.inf,

    "2-gram_ice_incidence_estimate_d0": 81,
    "2-gram_ice_modified_incidence_estimate_d0": 81,
    "2-gram_chao2_incidence_estimate_d0": 81,
    "2-gram_iChao2_incidence_estimate_d0": 81,
    "2-gram_jackknife1_incidence_estimate_d0": 81,
    "2-gram_jackknife2_incidence_estimate_d0": 81,
    "2-gram_tv_incidence_based_d0": math.inf,

    "3-gram_ace_abundance_estimate_d0": 729,
    "3-gram_ace_modified_abundance_estimate_d0": 729,
    "3-gram_chao1_abundance_estimate_d0": 729,
    "3-gram_iChao1_abundance_estimate_d0": 729,
    "3-gram_jackknife1_abundance_estimate_d0": 729,
    "3-gram_jackknife2_abundance_estimate_d0": 729,
    "3-gram_tv_abundance_based_d0": math.inf,

    "3-gram_ice_incidence_estimate_d0": 729,
    "3-gram_ice_modified_incidence_estimate_d0": 729,
    "3-gram_chao2_incidence_estimate_d0": 729,
    "3-gram_iChao2_incidence_estimate_d0": 729,
    "3-gram_jackknife1_incidence_estimate_d0": 729,
    "3-gram_jackknife2_incidence_estimate_d0": 729,
    "3-gram_tv_incidence_based_d0": math.inf,

    "4-gram_ace_abundance_estimate_d0": 6561,
    "4-gram_ace_modified_abundance_estimate_d0": 6561,
    "4-gram_chao1_abundance_estimate_d0": 6561,
    "4-gram_iChao1_abundance_estimate_d0": 6561,
    "4-gram_jackknife1_abundance_estimate_d0": 6561,
    "4-gram_jackknife2_abundance_estimate_d0": 6561,
    "4-gram_tv_abundance_based_d0": math.inf,

    "4-gram_ice_incidence_estimate_d0": 6561,
    "4-gram_ice_modified_incidence_estimate_d0": 6561,
    "4-gram_chao2_incidence_estimate_d0": 6561,
    "4-gram_iChao2_incidence_estimate_d0": 6561,
    "4-gram_jackknife1_incidence_estimate_d0": 6561,
    "4-gram_jackknife2_incidence_estimate_d0": 6561,
    "4-gram_tv_incidence_based_d0": math.inf,

    "5-gram_ace_abundance_estimate_d0": 59049,
    "5-gram_ace_modified_abundance_estimate_d0": 59049,
    "5-gram_chao1_abundance_estimate_d0": 59049,
    "5-gram_iChao1_abundance_estimate_d0": 59049,
    "5-gram_jackknife1_abundance_estimate_d0": 59049,
    "5-gram_jackknife2_abundance_estimate_d0": 59049,
    "5-gram_tv_abundance_based_d0": math.inf,

    "5-gram_ice_incidence_estimate_d0": 59049,
    "5-gram_ice_modified_incidence_estimate_d0": 59049,
    "5-gram_chao2_incidence_estimate_d0": 59049,
    "5-gram_iChao2_incidence_estimate_d0": 59049,
    "5-gram_jackknife1_incidence_estimate_d0": 59049,
    "5-gram_jackknife2_incidence_estimate_d0": 59049,
    "5-gram_tv_incidence_based_d0": math.inf
}

true_values_net_7 = {
    "1-gram_abundance_estimate_d0": 9,
    "1-gram_abundance_estimate_d1": 7.70,
    "1-gram_abundance_estimate_d2": 6.63,
    "1-gram_incidence_estimate_d0": 9,
    "1-gram_incidence_estimate_d1": 8.36,
    "1-gram_incidence_estimate_d2": 7.81,
    "2-gram_abundance_estimate_d0": 18,
    "2-gram_abundance_estimate_d1": 15.66,
    "2-gram_abundance_estimate_d2": 12.90,
    "2-gram_incidence_estimate_d0": 18,
    "2-gram_incidence_estimate_d1": 15.86,
    "2-gram_incidence_estimate_d2": 14.39,
    "3-gram_abundance_estimate_d0": 33,
    "3-gram_abundance_estimate_d1": 26.92,
    "3-gram_abundance_estimate_d2": 22.85,
    "3-gram_incidence_estimate_d0": 33,
    "3-gram_incidence_estimate_d1": 27.91,
    "3-gram_incidence_estimate_d2": 24.73,
    "4-gram_abundance_estimate_d0": 55,
    "4-gram_abundance_estimate_d1": 43.10,
    "4-gram_abundance_estimate_d2": 34.84,
    "4-gram_incidence_estimate_d0": 55,
    "4-gram_incidence_estimate_d1": 44.10,
    "4-gram_incidence_estimate_d2": 36.42,
    "5-gram_abundance_estimate_d0": 85,
    "5-gram_abundance_estimate_d1": 66.72,
    "5-gram_abundance_estimate_d2": 54.35,
    "5-gram_incidence_estimate_d0": 85,
    "5-gram_incidence_estimate_d1": 67.90,
    "5-gram_incidence_estimate_d2": 56.40,
    "tv_abundance_estimate_d0": math.inf,
    "tv_abundance_estimate_d1": math.inf,
    "tv_abundance_estimate_d2": math.inf,
    "tv_incidence_estimate_d0": math.inf,
    "tv_incidence_estimate_d1": math.inf,
    "tv_incidence_estimate_d2": math.inf,

    "1-gram_ace_abundance_estimate_d0": 9,
    "1-gram_ace_modified_abundance_estimate_d0": 9,
    "1-gram_chao1_abundance_estimate_d0": 9,
    "1-gram_iChao1_abundance_estimate_d0": 9,
    "1-gram_jackknife1_abundance_estimate_d0": 9,
    "1-gram_jackknife2_abundance_estimate_d0": 9,
    "1-gram_tv_abundance_based_d0": math.inf,

    "1-gram_ice_incidence_estimate_d0": 9,
    "1-gram_ice_modified_incidence_estimate_d0": 9,
    "1-gram_chao2_incidence_estimate_d0": 9,
    "1-gram_iChao2_incidence_estimate_d0": 9,
    "1-gram_jackknife1_incidence_estimate_d0": 9,
    "1-gram_jackknife2_incidence_estimate_d0": 9,
    "1-gram_tv_incidence_based_d0": math.inf,

    "2-gram_ace_abundance_estimate_d0": 18,
    "2-gram_ace_modified_abundance_estimate_d0": 18,
    "2-gram_chao1_abundance_estimate_d0": 18,
    "2-gram_iChao1_abundance_estimate_d0": 18,
    "2-gram_jackknife1_abundance_estimate_d0": 18,
    "2-gram_jackknife2_abundance_estimate_d0": 18,
    "2-gram_tv_abundance_based_d0": math.inf,

    "2-gram_ice_incidence_estimate_d0": 18,
    "2-gram_ice_modified_incidence_estimate_d0": 18,
    "2-gram_chao2_incidence_estimate_d0": 18,
    "2-gram_iChao2_incidence_estimate_d0": 18,
    "2-gram_jackknife1_incidence_estimate_d0": 18,
    "2-gram_jackknife2_incidence_estimate_d0": 18,
    "2-gram_tv_incidence_based_d0": math.inf,

    "3-gram_ace_abundance_estimate_d0": 33,
    "3-gram_ace_modified_abundance_estimate_d0": 33,
    "3-gram_chao1_abundance_estimate_d0": 33,
    "3-gram_iChao1_abundance_estimate_d0": 33,
    "3-gram_jackknife1_abundance_estimate_d0": 33,
    "3-gram_jackknife2_abundance_estimate_d0": 33,
    "3-gram_tv_abundance_based_d0": math.inf,

    "3-gram_ice_incidence_estimate_d0": 33,
    "3-gram_ice_modified_incidence_estimate_d0": 33,
    "3-gram_chao2_incidence_estimate_d0": 33,
    "3-gram_iChao2_incidence_estimate_d0": 33,
    "3-gram_jackknife1_incidence_estimate_d0": 33,
    "3-gram_jackknife2_incidence_estimate_d0": 33,
    "3-gram_tv_incidence_based_d0": math.inf,

    "4-gram_ace_abundance_estimate_d0": 55,
    "4-gram_ace_modified_abundance_estimate_d0": 55,
    "4-gram_chao1_abundance_estimate_d0": 55,
    "4-gram_iChao1_abundance_estimate_d0": 55,
    "4-gram_jackknife1_abundance_estimate_d0": 55,
    "4-gram_jackknife2_abundance_estimate_d0": 55,
    "4-gram_tv_abundance_based_d0": math.inf,

    "4-gram_ice_incidence_estimate_d0": 55,
    "4-gram_ice_modified_incidence_estimate_d0": 55,
    "4-gram_chao2_incidence_estimate_d0": 55,
    "4-gram_iChao2_incidence_estimate_d0": 55,
    "4-gram_jackknife1_incidence_estimate_d0": 55,
    "4-gram_jackknife2_incidence_estimate_d0": 55,
    "4-gram_tv_incidence_based_d0": math.inf,

    "5-gram_ace_abundance_estimate_d0": 85,
    "5-gram_ace_modified_abundance_estimate_d0": 85,
    "5-gram_chao1_abundance_estimate_d0": 85,
    "5-gram_iChao1_abundance_estimate_d0": 85,
    "5-gram_jackknife1_abundance_estimate_d0": 85,
    "5-gram_jackknife2_abundance_estimate_d0": 85,
    "5-gram_tv_abundance_based_d0": math.inf,

    "5-gram_ice_incidence_estimate_d0": 85,
    "5-gram_ice_modified_incidence_estimate_d0": 85,
    "5-gram_chao2_incidence_estimate_d0": 85,
    "5-gram_iChao2_incidence_estimate_d0": 85,
    "5-gram_jackknife1_incidence_estimate_d0": 85,
    "5-gram_jackknife2_incidence_estimate_d0": 85,
    "5-gram_tv_incidence_based_d0": math.inf

}

true_values_net_8 = {
    "1-gram_abundance_estimate_d0": 9,
    "1-gram_abundance_estimate_d1": 9,
    "1-gram_abundance_estimate_d2": 9,
    "1-gram_incidence_estimate_d0": 9,
    "1-gram_incidence_estimate_d1": 9,
    "1-gram_incidence_estimate_d2": 9,
    "2-gram_abundance_estimate_d0": 72,
    "2-gram_abundance_estimate_d1": 72,
    "2-gram_abundance_estimate_d2": 72,
    "2-gram_incidence_estimate_d0": 72,
    "2-gram_incidence_estimate_d1": 72,
    "2-gram_incidence_estimate_d2": 72,
    "3-gram_abundance_estimate_d0": 504,
    "3-gram_abundance_estimate_d1": 504,
    "3-gram_abundance_estimate_d2": 504,
    "3-gram_incidence_estimate_d0": 504,
    "3-gram_incidence_estimate_d1": 504,
    "3-gram_incidence_estimate_d2": 504,
    "4-gram_abundance_estimate_d0": 3024,
    "4-gram_abundance_estimate_d1": 3024,
    "4-gram_abundance_estimate_d2": 3024,
    "4-gram_incidence_estimate_d0": 3024,
    "4-gram_incidence_estimate_d1": 3024,
    "4-gram_incidence_estimate_d2": 3024,
    "5-gram_abundance_estimate_d0": 15120,
    "5-gram_abundance_estimate_d1": 15120,
    "5-gram_abundance_estimate_d2": 15120,
    "5-gram_incidence_estimate_d0": 15120,
    "5-gram_incidence_estimate_d1": 15120,
    "5-gram_incidence_estimate_d2": 15120,
    "tv_abundance_estimate_d0": 362880,
    "tv_abundance_estimate_d1": 362880,
    "tv_abundance_estimate_d2": 362880,
    "tv_incidence_estimate_d0": 362880,
    "tv_incidence_estimate_d1": 362880,
    "tv_incidence_estimate_d2": 362880,


    # true values for abundance- & incidence-based estimators
   "1-gram_ace_abundance_estimate_d0": 9,
   "1-gram_ace_modified_abundance_estimate_d0": 9,
   "1-gram_iChao1_abundance_estimate_d0": 9,
   "1-gram_jackknife1_abundance_estimate_d0": 9,
   "1-gram_jackknife2_abundance_estimate_d0": 9,
   "1-gram_tv_abundance_based_d0": 362880,

   "1-gram_ice_incidence_estimate_d0": 9,
   "1-gram_ice_modified_incidence_estimate_d0": 9,
   "1-gram_iChao2_incidence_estimate_d0": 9,
   "1-gram_jackknife1_incidence_estimate_d0": 9,
   "1-gram_jackknife2_incidence_estimate_d0": 9,
   "1-gram_tv_incidence_based_d0": 362880,

   "2-gram_ace_abundance_estimate_d0": 72,
   "2-gram_ace_modified_abundance_estimate_d0": 72,
   "2-gram_iChao1_abundance_estimate_d0": 72,
   "2-gram_jackknife1_abundance_estimate_d0": 72,
   "2-gram_jackknife2_abundance_estimate_d0": 72,
   "2-gram_tv_abundance_based_d0": 362880,

   "2-gram_ice_incidence_estimate_d0": 72,
   "2-gram_ice_modified_incidence_estimate_d0": 72,
   "2-gram_iChao2_incidence_estimate_d0": 72,
   "2-gram_jackknife1_incidence_estimate_d0": 72,
   "2-gram_jackknife2_incidence_estimate_d0": 72,
   "2-gram_tv_incidence_based_d0": 362880,

   "3-gram_ace_abundance_estimate_d0": 504,
   "3-gram_ace_modified_abundance_estimate_d0": 504,
   "3-gram_iChao1_abundance_estimate_d0": 504,
   "3-gram_jackknife1_abundance_estimate_d0": 504,
   "3-gram_jackknife2_abundance_estimate_d0": 504,
   "3-gram_tv_abundance_based_d0": 362880,

   "3-gram_ice_incidence_estimate_d0": 504,
   "3-gram_ice_modified_incidence_estimate_d0": 504,
   "3-gram_iChao2_incidence_estimate_d0": 504,
   "3-gram_jackknife1_incidence_estimate_d0": 504,
   "3-gram_jackknife2_incidence_estimate_d0": 504,
   "3-gram_tv_incidence_based_d0": 362880,

   "4-gram_ace_abundance_estimate_d0": 3024,
   "4-gram_ace_modified_abundance_estimate_d0": 3024,
   "4-gram_iChao1_abundance_estimate_d0": 3024,
   "4-gram_jackknife1_abundance_estimate_d0": 3024,
   "4-gram_jackknife2_abundance_estimate_d0": 3024,
   "4-gram_tv_abundance_based_d0": 362880,

   "4-gram_ice_incidence_estimate_d0": 3024,
   "4-gram_ice_modified_incidence_estimate_d0": 3024,
   "4-gram_iChao2_incidence_estimate_d0": 3024,
   "4-gram_jackknife1_incidence_estimate_d0": 3024,
   "4-gram_jackknife2_incidence_estimate_d0": 3024,
   "4-gram_tv_incidence_based_d0": 362880,

   "5-gram_ace_abundance_estimate_d0": 15120,
   "5-gram_ace_modified_abundance_estimate_d0": 15120,
   "5-gram_iChao1_abundance_estimate_d0": 15120,
   "5-gram_jackknife1_abundance_estimate_d0": 15120,
   "5-gram_jackknife2_abundance_estimate_d0": 15120,
   "5-gram_tv_abundance_based_d0": 362880,

   "5-gram_ice_incidence_estimate_d0": 15120,
   "5-gram_ice_modified_incidence_estimate_d0": 15120,
   "5-gram_iChao2_incidence_estimate_d0": 15120,
   "5-gram_jackknife1_incidence_estimate_d0": 15120,
   "5-gram_jackknife2_incidence_estimate_d0": 15120,
   "5-gram_tv_incidence_based_d0": 362880

}

#Used in Evaluation
# MODEL 3
#evaluate_model("./nets/net_3.pnml", "baseline_eval_model_3", 200, 1000, true_values_net_3)
evaluate_model("./nets/net_3.pnml", "baseline_eval_model_3", 200, 200, true_values_net_3) # limit sample size to 200

# MODEL 7
#evaluate_model("./nets/net_7.pnml", "baseline_eval_model_7", 200, 1000, true_values_net_7)
#evaluate_model("./nets/net_7.pnml", "baseline_eval_model_7", 200, 200, true_values_net_7) # limit: 200 samples

# MODEL 8
#evaluate_model("./nets/net_8.pnml", "baseline_eval_model_8", 200, 5000, true_values_net_8)
#evaluate_model("./nets/net_8.pnml", "baseline_eval_model_8", 200, 200, true_values_net_8) # limit: 200 samples

#Other nets
#evaluate_model("./nets/model_1.pnml", "baseline_eval_model_1", 100, 2000)
#evaluate_model("./nets/model_2.pnml", "baseline_eval_model_2", 100, 2000)
#evaluate_model("./nets/model_4.pnml", "baseline_eval_model_4", 100, 1000)
#evaluate_model("./nets/model_5.pnml", "baseline_eval_model_5", 100, 1000)
#evaluate_model("./nets/model_6.pnml", "baseline_eval_model_6", 100, 1000)



