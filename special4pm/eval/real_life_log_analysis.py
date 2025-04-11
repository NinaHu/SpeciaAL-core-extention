from functools import partial
import pm4py
from matplotlib import pyplot as plt

from special4pm.visualization.visualization import plot_diversity_profile, plot_completeness_profile, \
    plot_rank_abundance, plot_expected_sampling_effort

from special4pm.estimation import SpeciesEstimator
from special4pm.species import retrieve_species_n_gram, species_retrieval


def init_estimator(step_size):
    estimator = SpeciesEstimator(step_size=step_size, l_n=[0.99,0.95,0.90,0.80])
    estimator.register("1-gram", partial(retrieve_species_n_gram, n=1))
    estimator.register("2-gram", partial(retrieve_species_n_gram, n=2))
    estimator.register("tv", species_retrieval.retrieve_species_trace_variant)
    estimator.register("t1", partial(species_retrieval.retrieve_timed_activity, interval_size=1))
    estimator.register("t5", partial(species_retrieval.retrieve_timed_activity, interval_size=5))
    estimator.register("t30", partial(species_retrieval.retrieve_timed_activity, interval_size=30))
    estimator.register("t_e", species_retrieval.retrieve_timed_activity_exponential)

    return estimator


def profile_log(log, name):
    log = pm4py.read_xes(log, return_legacy_log_object=True)
    print(len(log))
    step_size = int(len(log) / 200)
    estimator = init_estimator(step_size=step_size)
    estimator.apply(log, verbose=True)
    estimator.add_bootstrap_ci(200)

    estimator.to_dataFrame().to_csv("out/" + name + ".csv", index=False)
    estimator.to_dataFrame(include_all=False).to_csv("out/" + name + "_final_only.csv", index=False)

    for species in estimator.metrics.keys():
        plot_rank_abundance(estimator, species, abundance=False, save_to="fig/" + name + "_" + species + "_rank_abundance.pdf")
        plot_diversity_profile(estimator, species, abundance=False, save_to="fig/" + name + "_diversity_profile.pdf")
        plot_completeness_profile(estimator, species, abundance=False, save_to="fig/" + name + "_completeness_profile.pdf")
        plot_expected_sampling_effort(estimator, species, abundance=False, save_to="fig/" + name + "_effort.pdf")


profile_log("../../logs/BPI_Challenge_2012.xes", "real_eval_BPI 2012")

profile_log("../../logs/Road_Traffic_Fines_Management_Process.xes", "real_eval_Road Traffic Fines")

profile_log("../../logs/Sepsis_Cases_-_Event_Log.xes", "real_eval_Sepsis Cases")

profile_log("../../logs/BPI_Challenge_2018.xes", "real_eval_BPI 2018")

profile_log("../../logs/BPI_Challenge_2019.xes", "real_eval_BPI 2019")