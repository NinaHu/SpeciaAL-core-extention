import copy
import math
from cProfile import label
from functools import partial

import pandas as pd
import pm4py
from matplotlib import pyplot as plt

from special4pm.estimation import SpeciesEstimator
from special4pm.simulation.simulation import simulate_model
from special4pm.visualization.visualization import plot_diversity_profile, plot_expected_sampling_effort, \
    plot_completeness_profile, plot_rank_abundance
from tqdm import tqdm

from special4pm.species import retrieve_species_n_gram, species_retrieval


def init_estimator(step_size, species_id):
    estimator = SpeciesEstimator(step_size=step_size)
    if species_id == "1-gram" or species_id == "all":
        estimator.register("1-gram", partial(retrieve_species_n_gram, n=1))
    if species_id == "2-gram" or species_id == "all":
        estimator.register("2-gram", partial(retrieve_species_n_gram, n=2))
    if species_id == "3-gram" or species_id == "all":
        estimator.register("3-gram", partial(retrieve_species_n_gram, n=3))
    if species_id == "4-gram" or species_id == "all":
        estimator.register("4-gram", partial(retrieve_species_n_gram, n=4))
    if species_id == "5-gram" or species_id == "all":
        estimator.register("5-gram", partial(retrieve_species_n_gram, n=5))
    return estimator


def profile_model(model, i, f, log, name):
    estimators = []
    estimator = init_estimator(None, "all")
    estimator.apply(log)

    model_est = init_estimator(None, "all")
    for x in range(100):
        print(name,(x+1)*(int(len(log)/100)))
        sim_log = simulate_model(model, i, f, int(len(log)/100), log)
        model_est.apply(sim_log, verbose=False)

    for species in estimator.metrics.keys():
        plot_rank_abundance(estimator, species, abundance=False, save_to="fig/" + name + "_rank_abundance.pdf")
        plot_diversity_profile(estimator, species, abundance=False,
                               save_to="fig/" + name + "_" + species + "_diversity_profile.pdf")
        plot_completeness_profile(estimator, species, abundance=False,
                                  save_to="fig/" + name + "_" + species + "_completeness_profile.pdf")
        plot_expected_sampling_effort(estimator, species, abundance=False,
                                      save_to="fig/" + name + "_" + species + "_effort.pdf")

    df = model_est.to_dataFrame(include_all=False).to_csv("out/" + name + ".csv", index=False)
    return

    #print("Simulating initial Sample of 80% Size")
    #sim_base_log = simulate_model(model, i, f, int(len(log) * 0.8), log)

    #simulate event log of same size as initial event log
    for species_id in ["1-gram", "2-gram", "3-gram", "4-gram", "5-gram"]:
        print("Beginning Analysis of "+species_id)
        base_coverage = estimator.metrics[species_id]["incidence_c1"][-1]

        #sim_log = simulate_model(model, i, f, int(len(log) * 0.1), log)
        model_est = init_estimator(None, species_id)
        model_est.apply(sim_log, verbose=False)
        model_coverage = model_est.metrics[species_id]["incidence_c1"][-1]
        # print("Baseline Log - " + species_id + " : " +str(base_coverage))

        #iterations = 100
        #while base_coverage - model_coverage > 0.05:
        #    sim_log = simulate_model(model, i, f, iterations, log)
        #    model_est.apply(sim_log, verbose=False)
        #    model_coverage = model_est.metrics[species_id]["incidence_c1"][-1]
        #    print(name, species_id, model_est.metrics[species_id]["incidence_no_observations"][-1], str(abs(model_coverage - base_coverage)), str(base_coverage), str(model_coverage))
        #    iterations = iterations * 2

        #TODO add bootstrap intervals? these take aaaages
        #model_est.add_bootstrap_ci(100)
        estimators.append(model_est)
        #estimator.to_dataFrame().to_csv("out/" + name + ".csv", index=False)
        #estimator.to_dataFrame(include_all=False).to_csv("out/" + name + "_final_only.csv", index=False)

        for species in estimator.metrics.keys():
            plot_rank_abundance(estimator, species, abundance=False, save_to="fig/" + name + "_rank_abundance.pdf")
            plot_diversity_profile(estimator, species, abundance=False,
                                   save_to="fig/" + name + "_" + species + "_diversity_profile.pdf")
            plot_completeness_profile(estimator, species, abundance=False,
                                      save_to="fig/" + name + "_" + species +"_completeness_profile.pdf")
            plot_expected_sampling_effort(estimator, species, abundance=False, save_to="fig/" + name + "_" + species +"_effort.pdf")

    df = pd.concat([x.to_dataFrame(include_all=False) for x in estimators])
    df.to_csv("out/" + name + ".csv", index=False)

logs = [
    #("../../logs/Sepsis_Cases_-_Event_Log.xes", "SEPSIS"),              #ILP missing
    ####("../../logs/Road_Traffic_Fines_Management_Process.xes", "RTF"),    #ILP missing
    #("../../logs/BPI_Challenge_2012.xes", "BPI2012"),                   #ILP missing
    #("../../logs/BPI_Challenge_2018.xes", "BPI2018"),                    #
    ("../../logs/BPI_Challenge_2019.xes", "BPI2019")
]

for log_path, log_name in logs:
    log = pm4py.read_xes(log_path, return_legacy_log_object=True)

#    print(log_name +": 1. Alpha")
#    mod, i, f = pm4py.discover_petri_net_alpha(log)
#    profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_alpha")

#    print(log_name +": 2. Inductive Miner")
#    mod, i, f = pm4py.discover_petri_net_inductive(log)
#    profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_inductive_0.0")

#    print(log_name +": 3. Inductive Miner 0.2")
#    mod, i, f = pm4py.discover_petri_net_inductive(log, noise_threshold=0.2)
#    profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_inductive_0.2")

    #print(log_name +": 4. Inductive Miner 0.4")
    #mod, i, f = pm4py.discover_petri_net_inductive(log, noise_threshold=0.4)
    #profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_inductive_0.4")

    #print(log_name +": 5. Inductive Miner 0.6")
    #mod, i, f = pm4py.discover_petri_net_inductive(log, noise_threshold=0.6)
    #profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_inductive_0.6")

#    print(log_name +": 6. Inductive Miner 0.8")
#    mod, i, f = pm4py.discover_petri_net_inductive(log, noise_threshold=0.8)
#    profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_inductive_0.8")

    print(log_name +": 7. ILP 1.0")
    mod, i, f = pm4py.discover_petri_net_ilp(log, alpha=1.0)
    profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_ILP_1.0")

    print(log_name +": 8. ILP 0.8")
    mod, i, f = pm4py.discover_petri_net_ilp(log, alpha=0.8)
    profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_ILP_0.8")

    #print(log_name +": 9. ILP 0.6")
    #mod, i, f = pm4py.discover_petri_net_ilp(log, alpha=0.6)
    #profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_ILP_0.6")

    #print(log_name +": 10. ILP 0.4")
    #mod, i, f = pm4py.discover_petri_net_ilp(log, alpha=0.4)
    #profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_ILP_0.4")

    print(log_name +": 11. ILP 0.2")
    mod, i, f = pm4py.discover_petri_net_ilp(log, alpha=0.2)
    profile_model(mod, i, f, log, "log_vs_model_eval_"+log_name+"_ILP_0.2")
