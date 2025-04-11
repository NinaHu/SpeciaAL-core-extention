from functools import partial

import pm4py

from special4pm.estimation import SpeciesEstimator
from special4pm.species import retrieve_species_n_gram
from special4pm.visualization import plot_rank_abundance, plot_completeness_profile, plot_diversity_profile

#Estimates completeness profiles and species richness for different species retrieval functions on the provided
#log, printing obtained metrics over all observations in the log each 100 traces

#Estimating species richness of an event log
#PATH_TO_XES = "logs/BPI_Challenge_2018.xes" # Log: Martin
PATH_TO_XES = "./logs/Road_Traffic_Fine_Management_Process.xes" # Log: Nina
log = pm4py.read_xes(PATH_TO_XES, return_legacy_log_object=True)


estimator = SpeciesEstimator(step_size=None)
estimator.register("1-gram", partial(retrieve_species_n_gram, n=1))
estimator.register("2-gram", partial(retrieve_species_n_gram, n=2))
estimator.register("3-gram", partial(retrieve_species_n_gram, n=3))
estimator.register("4-gram", partial(retrieve_species_n_gram, n=4))
estimator.register("5-gram", partial(retrieve_species_n_gram, n=5))

estimator.apply(log)
estimator.print_metrics()
estimator.to_dataFrame().to_csv("test.csv", index=False)
estimator.summarize()

plot_rank_abundance(estimator, "2-gram", save_to="rank_abundance_curve.pdf")
plot_diversity_profile(estimator, "2-gram", save_to="diversity.pdf")
plot_completeness_profile(estimator, "2-gram", save_to="completeness.pdf")
