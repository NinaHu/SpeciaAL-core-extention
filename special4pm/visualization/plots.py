from functools import partial

import numpy as np
import pandas as pd
import pm4py
from matplotlib import pyplot as plt

import species_estimator
import species_retrieval
from rarefaction_extrapolation import rarefy_extrapolate_bootstrap_all

WIDTH = 9

# TODO CONFIDENCE DYNAMIC!!!!!!
def summarize_metric(metric, estimators):
    df_all = pd.DataFrame([getattr(x, metric) for x in estimators],
                          columns=[x for x in range(0, len(getattr(estimators[0], metric)))])
    df_stats = pd.concat([
        pd.DataFrame([df_all[x].min()] for x in df_all.columns),
        pd.DataFrame([df_all[x].nsmallest(2, keep='all').iloc[-1]] for x in df_all.columns),
        pd.DataFrame(df_all.median()),
        pd.DataFrame([df_all[x].nlargest(2, keep='all').iloc[-1]] for x in df_all.columns),
        pd.DataFrame([df_all[x].max()] for x in df_all.columns)]
        , axis=1)
    df_stats.columns = ["min", "lower_ci", "median", "upper_ci", "max"]
    return df_stats

def plot_hill_numbers_raripolated(estimation,abundance = True):
    if abundance:
        est, lci, uci, loc = rarefy_extrapolate_bootstrap_all(estimation, abundance_data = True, data_points=50,
                                                              bootstrap_repetitions=100)
        plt.fill_between(loc,lci[0],uci[0], alpha=0.4)
        plt.plot(loc,est[0])

        plt.fill_between(loc,lci[1],uci[1], alpha=0.4)
        plt.plot(loc,est[1])

        plt.fill_between(loc,lci[2],uci[2], alpha=0.4)
        plt.plot(loc,est[2])

        print(estimation.number_observations_incidence,estimation.number_observations_abundance)
        plt.plot(estimation.observation_ids_abundance,estimation.observed_species_count)
        plt.plot(estimation.observation_ids_abundance,estimation.D1_sample_abundance)
        plt.plot(estimation.observation_ids_abundance,estimation.D2_sample_abundance)
    else:
        est, lci, uci, loc = rarefy_extrapolate_bootstrap_all(estimation, abundance_data=False, data_points=50,
                                                              bootstrap_repetitions=100)
        print()
        print(estimation.observed_species_count)
        print(estimation.D1_sample_incidence)
        print(estimation.D2_sample_incidence)
        print(est[0])
        print(est[1])
        print(est[2])
        plt.fill_between(loc, lci[0], uci[0], alpha=0.4)
        plt.plot(loc, est[0])

        plt.fill_between(loc, lci[1], uci[1], alpha=0.4)
        plt.plot(loc, est[1])

        plt.fill_between(loc, lci[2], uci[2], alpha=0.4)
        plt.plot(loc, est[2])

        plt.plot(estimation.observation_ids, estimation.observed_species_count)
        plt.plot(estimation.observation_ids, estimation.D1_sample_incidence)
        plt.plot(estimation.observation_ids, estimation.D2_sample_incidence)
    plt.show()


def plot_rank_abundance(estimation, name):
    #plt.style.use('seaborn-v0_8-ticks')

    plt.rcParams['figure.figsize'] = [WIDTH, 3]
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    no_species = len(estimation.reference_sample_abundance.values())

    plt.plot(sorted(list(estimation.reference_sample_abundance.values()), reverse=True))
    plt.fill_between(np.linspace(0, no_species, no_species, endpoint=False),
                     sorted(list(estimation.reference_sample_abundance.values()), reverse=True),
                     [0 for _ in range(no_species)], alpha=0.4)
    #plt.title("Rank Abundance Curve", fontsize=22)
    plt.xlabel("Species Rank", fontsize=24)
    plt.ylabel("Occurrences", fontsize=24)
    plt.xticks([0, no_species-1], [1, no_species])
    plt.yticks([0, max(estimation.reference_sample_abundance.values())],
               [0, max(estimation.reference_sample_abundance.values())])

    plt.tight_layout()
    plt.savefig("figures/" + name + "_curve_abundance.pdf", format="pdf")
    plt.close()
    #plt.show()

    plt.rcParams['figure.figsize'] = [WIDTH, 3]
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    no_species = len(estimation.reference_sample_incidence.values())

    plt.plot(sorted(list(estimation.reference_sample_incidence.values()), reverse=True))
    plt.fill_between(np.linspace(0, no_species, no_species, endpoint=False),
                     sorted(list(estimation.reference_sample_incidence.values()), reverse=True),
                     [0 for _ in range(no_species)], alpha=0.4)
    #plt.title("Species Occurence (Incidence)", fontsize=22)
    plt.xlabel("Species Rank", fontsize=24)
    plt.ylabel("Occurrences", fontsize=24)
    plt.xticks([0, no_species-1], [1, no_species])
    plt.yticks([0, max(estimation.reference_sample_abundance.values()) ],
               [0, max(estimation.reference_sample_abundance.values()) ])

    plt.tight_layout()
    plt.savefig("figures/" + name + "_curve_incidence.pdf", format="pdf")
    plt.close()
    #plt.show()


# here we only show abundance plots - trends are equivalent for both sides
def plot_rank_abundances_multiple_estimators(names, estimations, est_id):
    names = list(names)

    plt.rcParams['figure.figsize'] = [2*WIDTH, 3]
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey="row")

    no_species = len(estimations[0].reference_sample_abundance.values())
    ax1.plot(sorted(list(estimations[0].reference_sample_abundance.values()), reverse=True))
    ax1.fill_between(np.linspace(0, no_species, no_species, endpoint=False),
                     sorted(list(estimations[0].reference_sample_abundance.values()), reverse=True),
                     [0 for _ in range(no_species)], alpha=0.5)
    ax1.set_title(names[0], fontsize=24)
    ax1.set_xlabel("Species Rank", fontsize=20)
    ax1.set_ylabel("Occurences", fontsize=20)
    ax1.set_xticks([0, no_species-1])
    ax1.set_xticklabels([1, no_species ])
    # ax1.set_yticks([0,max(estimations[0].reference_sample_abundance.values())-1],[0,max(estimations[0].reference_sample_abundance.values())-1])

    plt.yticks([0, max(estimations[0].reference_sample_abundance.values()) - 1],
               [0, max(estimations[0].reference_sample_abundance.values()) - 1])

    no_species = len(estimations[1].reference_sample_abundance.values())
    ax2.plot(sorted(list(estimations[1].reference_sample_abundance.values()), reverse=True))
    ax2.fill_between(np.linspace(0, no_species, no_species, endpoint=False),
                     sorted(list(estimations[1].reference_sample_abundance.values()), reverse=True),
                     [0 for _ in range(no_species)], alpha=0.5)
    ax2.set_title(names[1] + " Noise", fontsize=24)
    ax2.set_xlabel("Species Rank", fontsize=20)
    #ax2.set_ylabel("Occurences", fontsize=20)
    ax2.set_xticks([0, no_species-1])
    ax2.set_xticklabels([1, no_species ])
    # ax2.set_yticks([0,max(estimations[0].reference_sample_abundance.values())-1],[0,max(estimations[0].reference_sample_abundance.values())-1])

    no_species = len(estimations[2].reference_sample_abundance.values())
    ax3.plot(sorted(list(estimations[2].reference_sample_abundance.values()), reverse=True))
    ax3.fill_between(np.linspace(0, no_species, no_species, endpoint=False),
                     sorted(list(estimations[2].reference_sample_abundance.values()), reverse=True),
                     [0 for _ in range(no_species)], alpha=0.5)
    ax3.set_title(names[2] + " Noise", fontsize=24)
    ax3.set_xlabel("Species Rank", fontsize=20)
    #ax3.set_ylabel("Occurences", fontsize=20)
    ax3.set_xticks([0, no_species-1])
    ax3.set_xticklabels([1, no_species ])
    # ax3.set_yticks([0,max(estimations[0].reference_sample_abundance.values())-1],[0,max(estimations[0].reference_sample_abundance.values())-1])

    no_species = len(estimations[3].reference_sample_abundance.values())
    ax4.plot(sorted(list(estimations[3].reference_sample_abundance.values()), reverse=True))
    ax4.fill_between(np.linspace(0, no_species, no_species, endpoint=False),
                     sorted(list(estimations[3].reference_sample_abundance.values()), reverse=True),
                     [0 for _ in range(no_species)], alpha=0.5)
    ax4.set_title(names[3] + " Noise", fontsize=24)
    ax4.set_xlabel("Species Rank", fontsize=20)
    #ax4.set_ylabel("Occurences", fontsize=20)
    ax4.set_xticks([0, no_species-1])
    ax4.set_xticklabels([1, no_species])
    # ax4.set_yticks([0,max(estimations[0].reference_sample_abundance.values())-1],[0,max(estimations[0].reference_sample_abundance.values())-1])

    plt.tight_layout()
    plt.savefig("figures/noise_eval_" + est_id + ".pdf", format="pdf")
    plt.close()
    #plt.show()


def plot_all_stats(estimations, name, loc=[0, 20, 40, 60, 80, 100], labels=[0, 200, 400, 600, 800, 1000]):
    #plt.style.use('seaborn-v0_8-ticks')
    #plt.style.use('ggplot')

    #labels = estimations[0].observation_ids
    #print(labels)
    no_data_points = len(estimations[0].observation_ids)
    #no_repetitions = len(estimations)
    #no_observations_abundance = estimations[0].number_observations_abundance
    #no_observations_incidence = estimations[0].number_observations_incidence

    obs_stats = summarize_metric("observed_species_count", estimations)
    chao2_abundance_stats = summarize_metric("chao2_abundance", estimations)
    chao2_incidence_stats = summarize_metric("chao2_incidence", estimations)

    D1_abundance_sample_stats = summarize_metric("D1_sample_abundance", estimations)
    D1_incidence_sample_stats = summarize_metric("D1_sample_incidence", estimations)
    D1_abundance_estimated_stats = summarize_metric("D1_estimated_abundance", estimations)
    D1_incidence_estimated_stats = summarize_metric("D1_estimated_incidence", estimations)

    D2_abundance_sample_stats = summarize_metric("D2_sample_abundance", estimations)
    D2_incidence_sample_stats = summarize_metric("D2_sample_incidence", estimations)
    D2_abundance_estimated_stats = summarize_metric("D2_estimated_abundance", estimations)
    D2_incidence_estimated_stats = summarize_metric("D2_estimated_incidence", estimations)

    #singletons_abundance_stats = summarize_metric("Q1_abundance", estimations)
    #singletons_incidence_stats = summarize_metric("Q1_incidence", estimations)

    #doubletons_abundance_stats = summarize_metric("Q2_abundance", estimations)
    #doubletons_incidence_stats = summarize_metric("Q2_incidence", estimations)

    coverage_abundance_stats = summarize_metric("coverage_abundance", estimations)
    coverage_incidence_stats = summarize_metric("coverage_incidence", estimations)

    completeness_abundance_stats = summarize_metric("completeness_abundance", estimations)
    completeness_incidence_stats = summarize_metric("completeness_incidence", estimations)

    l90_abundance_stats = summarize_metric("l90_abundance", estimations)
    l90_incidence_stats = summarize_metric("l90_incidence", estimations)

    l95_abundance_stats = summarize_metric("l95_abundance", estimations)
    l95_incidence_stats = summarize_metric("l95_incidence", estimations)

    l99_abundance_stats = summarize_metric("l99_abundance", estimations)
    l99_incidence_stats = summarize_metric("l99_incidence", estimations)

    plt.rcParams['figure.figsize'] = [9, 5]
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    # abundance plots
    # NOTE: we only collect stats after trace level, thus we plot abundance-based stats using incidence observation counts
    # as number of observations instead of abundance observation counts

    # all hill numbers as observed, singletons vs doubletons, completeness vs coverage
    # for each hill number, actual value vs estimated value
    # f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey='col')
    # Hill number q=0 & q=1 & q=2
    # print(no_data_points, len(obs_stats["lower_ci"]), len(obs_stats["upper_ci"]))
    # print(list(obs_stats["lower_ci"]))
    # print(list(obs_stats["upper_ci"]))
    # #np.linspace(0, no_data_points, num=no_data_points).astype(float),
    # plt.fill_between(np.linspace(0, no_data_points, num=no_data_points), obs_stats["lower_ci"], obs_stats["upper_ci"],
    #                  alpha=0.4)
    # plt.plot(obs_stats["median"], label="q=0")
    # plt.fill_between(np.linspace(0, no_data_points, num=no_data_points),D1_abundance_sample_stats["lower_ci"],
    #                  D1_abundance_sample_stats["upper_ci"], alpha=0.4)
    # plt.plot(D1_abundance_sample_stats["median"], label="q=1")
    # plt.fill_between(np.linspace(0, no_data_points, num=no_data_points),D2_abundance_sample_stats["lower_ci"],
    #                  D2_abundance_sample_stats["upper_ci"], alpha=0.4)
    # plt.plot(D2_abundance_sample_stats["median"], label="q=2")
    # #plt.title("Sample-based Hill-number (Abundance)", fontsize=26)
    # plt.xlabel("Sample Size", fontsize=22)
    # plt.ylabel("Hill number", fontsize=22)
    # plt.xticks([0, 20, 40, 60, 80, 100], [0, 200, 400, 600, 800, 1000])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("figures/" + name + "_sample_hill_abundance.pdf", format="pdf")
    # #plt.close()
    # plt.show()

    # # Singletons vs Doubletons
    # plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), singletons_abundance_stats["lower_ci"],
    #                  singletons_abundance_stats["upper_ci"], alpha=0.4)
    # plt.plot(singletons_abundance_stats["median"], label="Singletons")
    # plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), doubletons_abundance_stats["lower_ci"],
    #                  doubletons_abundance_stats["upper_ci"], alpha=0.4)
    # plt.plot(doubletons_abundance_stats["median"], label="Doubletons")
    # #plt.title("Singletons vs Doubletons (Abundance)", fontsize=26)
    # plt.xlabel("Sample Size", fontsize=22)
    # plt.ylabel("Counts", fontsize=22)
    # plt.xticks([0, 20, 40, 60, 80, 100], [0, 200, 400, 600, 800, 1000])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("figures/" + name + "_q1_vs_q2_abundance.pdf", format="pdf")
    # plt.close()
    # #plt.show()

    # completeness vs Coverage
    plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), completeness_abundance_stats["lower_ci"],
                     completeness_abundance_stats["upper_ci"], alpha=0.4)
    plt.plot(completeness_abundance_stats["median"], label="Completeness")
    plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), coverage_abundance_stats["lower_ci"],
                     coverage_abundance_stats["upper_ci"], alpha=0.4)
    plt.plot(coverage_abundance_stats["median"], label="Coverage")
    #plt.title("Completeness & Coverage (Abundance)", fontsize=26)
    plt.xlabel("Sample Size", fontsize=24)
    plt.ylabel("Completeness", fontsize=24)
    plt.xticks(loc, labels)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("figures/" + name + "_completeness_vs_coverage_abundance.pdf", format="pdf")
    plt.close()
    #plt.show()

    # incidence-based stats
    # plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), obs_stats["lower_ci"], obs_stats["upper_ci"],
    #                  alpha=0.4)
    # plt.plot(obs_stats["median"], label="q=0")
    # plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), D1_incidence_sample_stats["lower_ci"],
    #                  D1_incidence_sample_stats["upper_ci"], alpha=0.4)
    # plt.plot(D1_incidence_sample_stats["median"], label="q=1")
    # plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), D2_incidence_sample_stats["lower_ci"],
    #                  D2_incidence_sample_stats["upper_ci"], alpha=0.4)
    # plt.plot(D2_incidence_sample_stats["median"], label="q=2")
    # #plt.title("Sample-based Hill-number (Incidence)", fontsize=26)
    # plt.xlabel("Sample Size", fontsize=24)
    # plt.ylabel("Hill number", fontsize=24)
    # plt.xticks([0, 20, 40, 60, 80, 100], [0, 200, 400, 600, 800, 1000])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("figures/" + name + "_sample_hill_incidence.pdf", format="pdf")
    # plt.close()
    # #plt.show()
    #
    # # Singletons vs Doubletons
    # plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), singletons_incidence_stats["lower_ci"],
    #                  singletons_incidence_stats["upper_ci"], alpha=0.4)
    # plt.plot(singletons_incidence_stats["median"], label="Singletons")
    # plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), doubletons_incidence_stats["lower_ci"],
    #                  doubletons_incidence_stats["upper_ci"], alpha=0.4)
    # plt.plot(doubletons_incidence_stats["median"], label="Doubletons")
    # #plt.title("Singletons vs Doubletons (Incidence)", fontsize=26)
    # plt.xlabel("Sample Size", fontsize=24)
    # plt.ylabel("Counts", fontsize=24)
    # plt.xticks([0, 20, 40, 60, 80, 100], [0, 200, 400, 600, 800, 1000])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("figures/" + name + "_q1_vs_q2_incidence.pdf", format="pdf")
    # plt.close()
    # #plt.show()

    # completeness vs Coverage
    plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), completeness_incidence_stats["lower_ci"],
                     completeness_incidence_stats["upper_ci"], alpha=0.4)
    plt.plot(completeness_incidence_stats["median"], label="Completeness")
    plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), coverage_incidence_stats["lower_ci"],
                     coverage_incidence_stats["upper_ci"], alpha=0.4)
    plt.plot(coverage_incidence_stats["median"], label="Coverage")
    #plt.title("Completeness & Coverage (Incidence)", fontsize=26)
    plt.xlabel("Sample Size", fontsize=24)
    plt.ylabel("Completeness", fontsize=24)
    plt.xticks(loc, labels)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("figures/" + name + "_completeness_vs_coverage_incidence.pdf", format="pdf")
    plt.close()
    #plt.show()



    plt.rcParams['figure.figsize'] = [3*WIDTH, 5]
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20

    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey='all')


    ax1.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), obs_stats["lower_ci"], obs_stats["upper_ci"],
                     alpha=0.4)
    ax1.plot(obs_stats["median"], label="Observed")
    ax1.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), chao2_abundance_stats["lower_ci"],
                     chao2_abundance_stats["upper_ci"], alpha=0.4)
    ax1.plot(chao2_abundance_stats["median"], label="Estimated")
    ax1.set_title("q=0", fontsize=28)
    ax1.set_xlabel("Sample Size", fontsize=24)
    ax1.set_ylabel("Hill number", fontsize=24)
    ax1.set_xticks(loc)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=20)

    ax2.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), D1_abundance_estimated_stats["lower_ci"],
                     D1_abundance_estimated_stats["upper_ci"], alpha=0.4)
    ax2.plot(D1_abundance_estimated_stats["median"], label="Observed")
    ax2.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), D1_abundance_sample_stats["lower_ci"],
                     D1_abundance_sample_stats["upper_ci"], alpha=0.4)
    ax2.plot(D1_abundance_sample_stats["median"], label="Estimated")
    ax2.set_title("q=1", fontsize=28)
    ax2.set_xlabel("Sample Size", fontsize=24)
    ax2.set_ylabel("Hill number", fontsize=24)
    ax2.set_xticks(loc)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=20)

    ax3.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), D2_abundance_estimated_stats["lower_ci"],
                     D2_abundance_estimated_stats["upper_ci"], alpha=0.4)
    ax3.plot(D2_abundance_estimated_stats["median"], label="Observed")
    ax3.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), D2_abundance_sample_stats["lower_ci"],
                     D2_abundance_sample_stats["upper_ci"], alpha=0.4)
    ax3.plot(D2_abundance_sample_stats["median"], label="Estimated")
    ax3.set_title("q=2", fontsize=28)
    ax3.set_xlabel("Sample Size", fontsize=24)
    ax3.set_ylabel("Hill number", fontsize=24)
    ax3.set_xticks(loc)
    ax3.set_xticklabels(labels)
    ax3.legend(fontsize=20)

    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("figures/" + name + "_estimations_abundance.pdf", format="pdf")
    plt.close()
    #plt.show()

    f, (ax4, ax5, ax6) = plt.subplots(nrows=1, ncols=3, sharey='all')

    ax4.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), obs_stats["lower_ci"], obs_stats["upper_ci"],
                     alpha=0.4)
    ax4.plot(obs_stats["median"], label="Observed")
    ax4.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), chao2_incidence_stats["lower_ci"],
                     chao2_incidence_stats["upper_ci"], alpha=0.4)
    ax4.plot(chao2_incidence_stats["median"], label="Estimated")
    ax4.set_title("q=0", fontsize=28)
    ax4.set_xlabel("Sample Size", fontsize=24)
    ax4.set_ylabel("Hill number", fontsize=24)
    ax4.set_xticks(loc)
    ax4.set_xticklabels(labels)
    ax4.legend(fontsize=20)

    ax5.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), D1_incidence_sample_stats["lower_ci"],
                     D1_incidence_sample_stats["upper_ci"], alpha=0.4)
    ax5.plot(D1_incidence_sample_stats["median"], label="Observed")
    ax5.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), D1_incidence_estimated_stats["lower_ci"],
                     D1_incidence_estimated_stats["upper_ci"], alpha=0.4)
    ax5.plot(D1_incidence_estimated_stats["median"], label="Estimated")
    ax5.set_title("q=1", fontsize=28)
    ax5.set_xlabel("Sample Size", fontsize=24)
    ax5.set_ylabel("Hill number", fontsize=24)
    ax5.set_xticks(loc)
    ax5.set_xticklabels(labels)
    ax5.legend(fontsize=20)

    ax6.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), D2_incidence_sample_stats["lower_ci"],
                     D2_incidence_sample_stats["upper_ci"], alpha=0.4)
    ax6.plot(D2_incidence_sample_stats["median"], label="Observed")
    ax6.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), D2_incidence_estimated_stats["lower_ci"],
                     D2_incidence_estimated_stats["upper_ci"], alpha=0.4)
    ax6.plot(D2_incidence_estimated_stats["median"], label="Estimated")
    ax6.set_title("q=2", fontsize=28)
    ax6.set_xlabel("Sample Size", fontsize=24)
    ax6.set_ylabel("Hill number", fontsize=24)
    ax6.set_xticks(loc)
    ax6.set_xticklabels(labels)
    ax6.legend(fontsize=20)

    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("figures/" + name + "_estimations_incidence.pdf", format="pdf")
    plt.close()
    #plt.show()

    plt.rcParams['figure.figsize'] = [WIDTH, 5]

    # f, (ax1,ax2) = plt.subplots(nrows=1, ncols=2)
    plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), l99_abundance_stats["lower_ci"],
                     l99_abundance_stats["upper_ci"], alpha=0.4)
    plt.plot(l99_abundance_stats["median"], label="l=.99")
    plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), l95_abundance_stats["lower_ci"],
                     l95_abundance_stats["upper_ci"], alpha=0.4)
    plt.plot(l95_abundance_stats["median"], label="l=.95")
    plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), l90_abundance_stats["lower_ci"],
                     l90_abundance_stats["upper_ci"], alpha=0.4)
    plt.plot(l90_abundance_stats["median"], label="l=.90")
    #plt.title("Expected Sampling Effort (Incidence)", fontsize=26)
    plt.xlabel("Sample Size", fontsize=24)
    plt.ylabel("Expected Sampling Effort", fontsize=24)
    plt.xticks(loc, labels)
    plt.legend(fontsize=20)

    plt.tight_layout()
    plt.savefig("figures/" + name + "_sampling_effort_abundance.pdf", format="pdf")
    plt.close()
    #plt.show()

    plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), l99_incidence_stats["lower_ci"],
                     l99_incidence_stats["upper_ci"], alpha=0.4)
    plt.plot(l99_incidence_stats["median"], label="l=.99")
    plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), l95_incidence_stats["lower_ci"],
                     l95_incidence_stats["upper_ci"], alpha=0.4)
    plt.plot(l95_incidence_stats["median"], label="l=.95")
    plt.fill_between(np.linspace(0, no_data_points, num=no_data_points).astype(float), l90_incidence_stats["lower_ci"],
                     l90_incidence_stats["upper_ci"], alpha=0.4)
    plt.plot(l90_incidence_stats["median"], label="l=.90")
    #plt.title("Expected Sampling Effort (Incidence)", fontsize=26)
    plt.xlabel("Sample Size", fontsize=24)
    plt.ylabel("Expected Sampling Effort", fontsize=24)
    plt.xticks(loc, labels)
    # plt.set_xticklabels()
    plt.legend(fontsize=20)

    plt.tight_layout()
    plt.savefig("figures/" + name + "_sampling_effort_incidence.pdf", format="pdf")
    plt.close()
    #plt.show()