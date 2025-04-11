import time
from functools import partial

import pm4py

from build.src.special4pm.species import species_retrieval
from special4pm.estimation import SpeciesEstimator
from special4pm.simulation.simulation import simulate_model
from special4pm.species import retrieve_species_n_gram


def init_estimator(step_size):
    estimator = SpeciesEstimator(step_size=step_size)
    estimator.register("1-gram", partial(retrieve_species_n_gram, n=1))
    estimator.register("2-gram", partial(retrieve_species_n_gram, n=2))
    estimator.register("3-gram", partial(retrieve_species_n_gram, n=3))
    estimator.register("4-gram", partial(retrieve_species_n_gram, n=4))
    estimator.register("5-gram", partial(retrieve_species_n_gram, n=5))
    estimator.register("tv", species_retrieval.retrieve_species_trace_variant)
    return estimator


def simulate(net, im, fm):
    return simulate_model(net, im, fm, 100)


def determine_hill_numbers(path, species_determining_stop):
    net, im, fm = pm4py.read_pnml(path)
    estimator = init_estimator(None)
    i=0
    while(True):
        i = i +100
        log = simulate(net, im, fm)
        estimator.apply(log)

        print("TRACE " + str(i))
        for species in estimator.metrics.keys():
            print(species + " D0: " + str(estimator.metrics[species]["abundance_sample_d0"][-1]), str(estimator.metrics[species]["incidence_sample_d0"][-1]))
            print(species + " D1: " + str(estimator.metrics[species]["abundance_sample_d1"][-1]), str(estimator.metrics[species]["incidence_sample_d1"][-1]))
            print(species + " D2: " + str(estimator.metrics[species]["abundance_sample_d2"][-1]), str(estimator.metrics[species]["incidence_sample_d2"][-1]))
            print(species + " C0: " + str(estimator.metrics[species]["abundance_c0"][-1]), str(estimator.metrics[species]["incidence_c0"][-1]))
            print(species + " C1: " + str(estimator.metrics[species]["abundance_c1"][-1]), str(estimator.metrics[species]["incidence_c1"][-1]))
            print()
        time.sleep(.2)
        #if (estimator.metrics[species_determining_stop]["abundance_c0"][-1]>=1.1):
        #    return

#determine_hill_numbers("../../nets/net_3.pnml", "5-gram")
#determine_hill_numbers("../../nets/net_7.pnml", "5-gram")
determine_hill_numbers("../../nets/net_8.pnml", "tv")

#determine_hill_numbers("../../nets/base_net.pnml", "5-gram")
#determine_hill_numbers("../../nets/net_added_e.pnml", "5-gram")
#determine_hill_numbers("../../nets/net_removed_b.pnml", "5-gram")
#determine_hill_numbers("../../nets/net_equal_prob.pnml", "5-gram")
