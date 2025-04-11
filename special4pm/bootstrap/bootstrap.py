import math
import statistics

import numpy as np

from special4pm.estimation.metrics import get_singletons, get_doubletons, get_total_species_count, \
    hill_number_asymptotic, completeness, \
    coverage


def generate_bootstrap_samples_abundance(reference_sample, n):
    s_obs = len(reference_sample)
    #print(reference_sample.values)
    sample_size = get_total_species_count(reference_sample)

    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)
    f_0 = 0
    if f_2>0:
        f_0 = ((sample_size-1)/sample_size) * f_1**2 / (2*f_2)
    else:
        f_0 = ((sample_size-1)/sample_size) * f_1 * (f_1-1) / 2
    f_0 = math.ceil(f_0)
    probabilities = []
    c = get_c_n_abundance(reference_sample, sample_size)
    factor = factor_abundance(reference_sample, sample_size, c)


    for x_i in reference_sample.values():
        adapted_p = (x_i / sample_size) * (1 - factor * ((1 - (x_i / sample_size)) ** sample_size))
        probabilities.append(adapted_p)
    [probabilities.append((1 - c) / f_0) for i in range(0, f_0)]

    rng = np.random.default_rng()
    bs_samples = []
    #for i in range(n):
    #    selected_species = choice([x for x in range(0, s_obs + f_0)], sample_size, p=probabilities)

    #    unique, counts = np.unique(selected_species, return_counts=True)
    for x in rng.multinomial(sample_size, probabilities, size=n):
        bs_samples.append( dict(zip(range(len(x)), x)))

    return bs_samples


def get_bootstrap_ci_incidence(reference_sample, sample_size, no_samples):
    samples = generate_bootstrap_samples_incidence(reference_sample, sample_size, no_samples)
    d0 = [hill_number_asymptotic(0, sample, sample_size, abundance=False) for sample in samples]
    d1 = [hill_number_asymptotic(1, sample, sample_size, abundance=False) for sample in samples]
    d2 = [hill_number_asymptotic(2, sample, sample_size, abundance=False) for sample in samples]
    c0 = [completeness(sample) for sample in samples]
    c1 = [coverage(sample, sample_size) for sample in samples]

    return [statistics.stdev(x)*1.96 for x in (d0,d1,d2,c0,c1)]


def generate_bootstrap_samples_incidence(reference_sample, sample_size, no_bs_samples):
    #get bootstrap distribution of species
    f_0 = 0

    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)

    if f_2>0:
        f_0 = ((sample_size-1)/sample_size) * f_1**2 / (2*f_2)
    else:
        f_0 = ((sample_size-1)/sample_size) * f_1 * (f_1-1) / 2
    f_0=math.ceil(f_0)

    probabilities = []
    c = get_c_n_incidence(reference_sample, sample_size)
    u = sum(reference_sample.values())

    factor = factor_incidence(reference_sample, sample_size, u, c)

    for y_i in reference_sample.values():
        adapted_p = (y_i / sample_size) * (1 - factor * (1 - (y_i / sample_size) ** sample_size))
        probabilities.append(adapted_p)
    [probabilities.append((u/sample_size) * (1 - c) / f_0) for i in range(0, f_0)]

    #generate all samples
    bootstrap_samples = [{} for _ in range(no_bs_samples)]
    #for n in range(0,sample_size):
    for s,p in enumerate(probabilities):
        species_counts = np.random.default_rng().binomial(n=sample_size, p=p, size=no_bs_samples)
        for x in range(no_bs_samples):
            if species_counts[x]==0:
                continue
            bootstrap_samples[x][s]=species_counts[x]

    return bootstrap_samples

def generate_bootstrap_estimates_incidence(reference_sample, sample_size):
    #get bootstrap distribution of species
    f_0 = 0

    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)

    if f_2>0:
        f_0 = ((sample_size-1)/sample_size) * f_1**2 / (2*f_2)
    else:
        f_0 = ((sample_size-1)/sample_size) * f_1 * (f_1-1) / 2
    f_0=math.ceil(f_0)

    probabilities = []
    c = get_c_n_incidence(reference_sample, sample_size)
    u = sum(reference_sample.values())

    factor = factor_incidence(reference_sample, sample_size, u, c)

    for y_i in reference_sample.values():
        adapted_p = (y_i / sample_size) * (1 - factor * (1 - (y_i / sample_size) ** sample_size))
        probabilities.append(adapted_p)
    [probabilities.append((u/sample_size) * (1 - c) / f_0) for i in range(0, f_0)]

    #generate all samples
    bootstrap_sample = {}
    #for n in range(0,sample_size):
    for s,p in enumerate(probabilities):
        species_count = np.random.default_rng().binomial(n=sample_size, p=p)
        bootstrap_sample[s]=species_count

    #collect estimates

    #return tuple
    return bootstrap_sample


def factor_abundance(reference_sample, n, c):
    if c == 1:
        return 0
    return (1 - c) / (sum([(x_i / n) * (1 - (x_i / n))**n for x_i in reference_sample.values()]))


def factor_incidence(reference_sample, n, u, c):
    if c == 1:
        return 0
    return ((u / n) * (1 - c)) / (sum([(x_i / n) * (1 - (x_i / n) ** n) for x_i in reference_sample.values()]))


def get_c_n_abundance(reference_sample, n):
    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)

    if f_2 > 0:
        return 1 - (f_1 / n) * (((n - 1) * f_1) / ((n - 1) * f_1 + 2 * f_2))
    else:
        return 1 - (f_1 / n) * (((n - 1) * (f_1 - 1)) / ((n - 1) * (f_1 - 1) + 2))


def get_c_n_incidence(reference_sample, n):
    u = sum(reference_sample.values())
    f_1 = get_singletons(reference_sample)
    f_2 = get_doubletons(reference_sample)
    if f_2 > 0:
        return 1-(f_1/ u) * (((n-1)*f_1)/((n-1)*f_1+2*f_2))
    else:
        return 1-(f_1/ u) * (((n-1)*(f_1-1))/((n-1)*(f_1-1)+2))