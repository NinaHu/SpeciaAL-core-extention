import copy
from functools import partial
import pm4py
from pm4py.objects.log.obj import EventLog
from special4pm.visualization.visualization import plot_diversity_profile, plot_completeness_profile, \
    plot_expected_sampling_effort

from build.src.special4pm.species import species_retrieval
from special4pm.estimation import SpeciesEstimator
from special4pm.species import retrieve_species_n_gram
from special4pm.visualization import plot_rank_abundance


def init_estimator(step_size):
    estimator = SpeciesEstimator(step_size=step_size)
    estimator.register("1-gram", partial(retrieve_species_n_gram, n=1))
    estimator.register("2-gram", partial(retrieve_species_n_gram, n=2))
    estimator.register("tv", species_retrieval.retrieve_species_trace_variant)
    estimator.register("t1", partial(species_retrieval.retrieve_timed_activity, interval_size=1))
    estimator.register("t30", partial(species_retrieval.retrieve_timed_activity, interval_size=30))
    estimator.register("t_e", species_retrieval.retrieve_timed_activity_exponential)
    return estimator


def profile_log(log, name):
    step_size = int(len(log) / 200)
    estimator = init_estimator(step_size=step_size)
    estimator.apply(log, verbose=True)
    estimator.add_bootstrap_ci(200)


    estimator.to_dataFrame().to_csv("out/" + name + ".csv", index=False)
    estimator.to_dataFrame(include_all=False).to_csv("out/" + name + "_final_only.csv", index=False)

    for species in estimator.metrics.keys():
        plot_rank_abundance(estimator, species, abundance=False, save_to="fig/" + name + "_rank_abundance.pdf")
        plot_diversity_profile(estimator, species, abundance=False, save_to="fig/" + name + "_diversity_profile.pdf")
        plot_completeness_profile(estimator, species, abundance=False,
                                  save_to="fig/" + name + "_completeness_profile.pdf")
        plot_expected_sampling_effort(estimator, species, abundance=False, save_to="fig/" + name + "_effort.pdf")


log = pm4py.read_xes("../../logs/Sepsis_Cases_-_Event_Log.xes", return_legacy_log_object=True)

log_pre_admission = copy.deepcopy(log)
log_post_admission = copy.deepcopy(log)

log_young = EventLog(attributes=log.attributes, extensions=log.extensions, omni_present=log.omni_present,
                     classifiers=log.classifiers, properties=log.properties)
log_old = EventLog(attributes=log.attributes, extensions=log.extensions, omni_present=log.omni_present,
                   classifiers=log.classifiers, properties=log.properties)

for t in range(0, len(log)):
    if log[t][0]['Age'] >= 60:
        log_old.append(log[t])
    else:
        log_young.append(log[t])
    for idx, e in enumerate(log[t]):
        if "Admission" in e["concept:name"]:
            if len(log_pre_admission[t][:idx + 1]) > 0:
                log_pre_admission[t] = log_pre_admission[t][:idx + 1]

            if len(log_post_admission[t][idx + 1:]) > 0:
                log_post_admission[t] = log_post_admission[t][idx + 1:]
            break

profile_log(log_pre_admission, "log_vs_log_eval_sepsis_cases_pre_admission")
profile_log(log_post_admission, "log_vs_log_eval_sepsis_cases_post_admission")
profile_log(log_young, "log_vs_log_eval_sepsis_cases_age_less_60")
profile_log(log_old, "log_vs_log_eval_sepsis_cases_age_geq_60")
