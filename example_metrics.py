from special4pm.estimation.metrics import estimate_species_richness_chao, estimate_simpson_diversity_incidence, \
    completeness, coverage, sampling_effort_incidence, estimate_exp_shannon_entropy_incidence

'''
Example script explaining how to directly calculate diversity and completeness profiles on observed species counts
'''
#We assume this to be Incidence Data
observed_species = {"A": 10, "B": 5, "C": 2, "D": 2, "E": 1, "F": 1}
number_observations = 10

est_species_richness = estimate_species_richness_chao(observed_species)
print("(D0) Asymptotic Species Richness:              " + str(estimate_species_richness_chao(observed_species)))
print("(D1) Asymptotic Exponential Shannon Entropy:   "+str(estimate_exp_shannon_entropy_incidence(observed_species, number_observations)))
print("(D2) Asymptotic Simpson Diversity:             "+str(estimate_simpson_diversity_incidence(observed_species, number_observations)))
print()
print("(C0) Completeness:                             "+str(completeness(observed_species)))
print("(C1) Coverage:                                 "+str(coverage(observed_species, number_observations)))
print("Additional Sampling Effort (l=.99):            "+str(sampling_effort_incidence(.99,observed_species, number_observations)))