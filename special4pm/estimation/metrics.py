import math
from random import sample

import mpmath
from numpy import euler_gamma
from scipy.special import digamma

from cachetools import cached

from collections import defaultdict, Counter # Nina


#TODO unify incidence and abundance-based methods in one function
def get_incidence_count(obs_species_counts: dict, i: int) -> int:
    """
    returns the number of species, that have an incidence count of i
    :param obs_species_counts: the species with corresponding incidence counts
    :param i: the incidence count
    :return: the number of species with incidence count i
    """
    return list(obs_species_counts.values()).count(i)


def get_singletons(obs_species_counts: dict) -> int:
    """
    returns the number of singletons species, i.e. those species that have an incidence count of 1
    :param obs_species_counts: the species with corresponding incidence counts
    :return: the number of species with incidence count 1
    """
    return list(obs_species_counts.values()).count(1)


def get_doubletons(obs_species_counts: dict) -> int:
    """
    returns the number of doubleton species, i.e. those species that have an incidence count of 2
    :param obs_species_counts: the species with corresponding incidence counts
    :return: the number of species with incidence count 2
    """
    return list(obs_species_counts.values()).count(2)


def get_number_observed_species(obs_species_counts: dict) -> int:
    """
    returns the number of observed species
    :param obs_species_counts: the species with corresponding incidence counts
    :return: the number of observed species
    """
    return len(obs_species_counts.keys())


def get_total_species_count(obs_species_counts):
    """
    returns the total number of species incidences, i.e. the sum of all species incidences in the reference sample
    :param obs_species_counts: the species with corresponding incidence counts
    :return: the sum of species incidences
    """
    return sum(obs_species_counts.values())


def hill_number(d: int, obs_species_counts: dict) -> float:
    """
    computes sample-based Hill number of order d for the reference sample.
    D=0 - species richness
    D=1 - Exponential of Shannon entropy
    D=2 - Simpson Diversity Index
    :param d: the order of the Hill number
    :param obs_species_counts: the species with corresponding incidence counts
    :return: the sample-based Hill number of order d
    """
    # sample-based species richness
    if d == 0:
        return get_number_observed_species(obs_species_counts)
    # sample-based exponential Shannon diversity
    if d == 1:
        return entropy_exp(obs_species_counts)
    # sample-based Simpson diversity
    if d == 2:
        return simpson_diversity(obs_species_counts)


def entropy_exp(obs_species_counts: dict) -> float:
    """
    computes the exponential of Shannon entropy
    :param obs_species_counts: the species with corresponding incidence counts
    :return: the exponential of Shannon entropy
    """
    total_species_count = get_total_species_count(obs_species_counts)
    # total = sum([obs_species_counts.values])
    return math.exp(-1 * sum(
        [x / total_species_count * math.log(x / total_species_count) for x in obs_species_counts.values()]))


def simpson_diversity(obs_species_counts: dict) -> float:
    """
    computes the Simpson diversity index
    :param obs_species_counts: the species with corresponding incidence counts
    :return: the Simpson diversity index
    """
    total_species_count = get_total_species_count(obs_species_counts)

    a = sum([(x / total_species_count) ** 2 for x in obs_species_counts.values()])
    # TODO check if return 1 is reasonable
    return a ** (1 / (1 - 2)) if a > 0 else 1


'''
Calculate asymptotic Hill number of order d for a reference sample
d=0 Species Richness
d=1 Exponential of Shannon Entropy
d=2 Simpson Diversity Index
'''


def hill_number_asymptotic(d: int, obs_species_counts: dict, sample_size: int, abundance: bool = True) -> float:
    """
    computes asymptotic Hill number of order d for the reference sample, for either abundance data or incidence data.
    D=0 - asymptotic species richness
    D=1 - asymptotic exponential of Shannon entropy
    D=2 - asymptotic Simpson diversity index
    :param d: the order of the Hill number
    :param obs_species_counts: the species with corresponding incidence counts
    :param sample_size: the sample size associated with the species incidence counts
    :param abundance: flag indicating the data type. Setting this 'True' indicates abundance-based data,
    setting this 'False' indicates incidence-based data
    :return: the asymptotic Hill number of order d
    """
    # asymptotic species richness
    # for species richness, there is no differentiation between abundance and incidence
    if d == 0:
        return estimate_species_richness_chao(obs_species_counts)
    # asymptotic Shannon entropy
    if d == 1:
        if abundance:
            return estimate_exp_shannon_entropy_abundance(obs_species_counts, sample_size)
        # incidence
        else:
            return estimate_exp_shannon_entropy_incidence(obs_species_counts, sample_size)
    # asymptotic Simpson diversity
    if d == 2:
        if abundance:
            return estimate_simpson_diversity_abundance(obs_species_counts, sample_size)
        # incidence
        else:
            return estimate_simpson_diversity_incidence(obs_species_counts, sample_size)


def estimate_species_richness_chao(obs_species_counts: dict) -> float:
    """
    computes the asymptotic(=estimated) species richness using the Chao1 estimator(for abundance data)
    or Chao2 estimator (for incidence data)
    :param obs_species_counts: the species with corresponding incidence counts
    :return: the estimated species richness
    """
    obs_species_count = get_number_observed_species(obs_species_counts)
    f_1 = get_singletons(obs_species_counts)
    f_2 = get_doubletons(obs_species_counts)

    if f_2 != 0:
        return obs_species_count + f_1 ** 2 / (2 * f_2)
    else:
        return obs_species_count + f_1 * (f_1 - 1) / 2


def estimate_species_richness_chao_corrected(obs_species_counts: dict) -> float:
    """
    computes the asymptotic(=estimated) species richness using the Chao1 estimator(for abundance data)
    or Chao2 estimator (for incidence data). Includes a correction term for very small sample sizes
    :param obs_species_counts: the species with corresponding incidence counts
    :return: the estimated species richness
    """
    obs_species_count = get_number_observed_species(obs_species_counts)
    f_1 = get_singletons(obs_species_counts)
    f_2 = get_doubletons(obs_species_counts)

    if f_2 != 0:
        return ((obs_species_count - 1) / obs_species_count) * obs_species_count + f_1 ** 2 / (2 * f_2)
    else:
        return ((obs_species_count - 1) / obs_species_count) * obs_species_count + f_1 * (f_1 - 1) / 2


### Nina

# Abundance-based dataset

def get_abundance_data():
    return [20, 15, 12, 9, 7, 2, 1, 1]


# incidence-based dataset
def get_species_counts_and_log():
    species_counts = [12, 12, 6, 5, 3, 4, 3, 2, 1]
    data_log = [
        ["A", "B", "C", "D"],
        ["A", "B", "E", "F"],
        ["B", "G", "H"],
        ["A", "C", "I", "G"],
        ["A", "B", "C", "D"],
        ["A", "B", "D", "C"],
        ["A", "B", "H", "C"],
        ["B", "A", "G"],
        ["A", "B", "E", "F"],
        ["A", "B", "C", "D"],
        ["A", "B", "F", "D"],
        ["E", "B", "F", "A"],
        ["A", "B"]
    ]
    return species_counts, data_log


# ACE Estimator
def calculate_C_ace (F1_abund, N_rare_abund):
    if N_rare_abund == 0:
        return 0
    return 1 - (F1_abund / N_rare_abund)


def calculate_gamma_sq_ace(S_rare_abund, C_ace, Fi_abund, N_rare_abund):
    if C_ace == 0 or N_rare_abund <= 1:
        # division by zero
        return 0

    #sum_term = sum((i + 1) * i * Fi_abund[i] for i in range(min(9, len(Fi_abund))))
    sum_term = sum(i * (i - 1) * Fi_abund[i - 1] for i in range(1, min(11, len(Fi_abund) + 1)))
    gamma_sq_ace = max(0, (S_rare_abund / C_ace) * (sum_term / (N_rare_abund * (N_rare_abund - 1))) - 1)
    return gamma_sq_ace


def ace(S_abund, S_rare_abund, F1_abund, N_rare_abund, Fi_abund):
    """
        Calculates ACE (Abundance-based Coverage Estimator) for species richness.

        Parameters:
        - S_abund: Count of abundant species (>10 individuals)
        - S_rare_abund: Count of rare species (≤10 individuals)
        - F1_abund: Number of singletons (species observed once)
        - N_rare_abund: Total count of individuals for rare species
        - Fi_abund: List of counts for species with exactly i individuals (i = 1 to 10)

        Returns:
        - S_ace: Estimated species richness (ACE result)
        - gamma_sq_ace: Variability of rare species
        """

    C_ace = calculate_C_ace(F1_abund, N_rare_abund)
    #print(f"C_ace: {C_ace}")

    gamma_sq_ace = calculate_gamma_sq_ace(S_rare_abund, C_ace, Fi_abund, N_rare_abund)

    #print(f"gamma_sq_ace: {gamma_sq_ace}")

    if C_ace == 0:
        return S_abund + S_rare_abund, gamma_sq_ace

    S_ace = S_abund + (S_rare_abund / C_ace) + (F1_abund / C_ace) * gamma_sq_ace
    return S_ace, gamma_sq_ace


def analyze_data_ace(data):
    data.reverse()
    # Calculate S_abund and N_rare_abund
    S_abund = sum(1 for x in data if x > 10)
    N_rare_abund = sum(x for x in data if x <= 10)
    F1_abund = sum(1 for x in data if x == 1)
    S_rare_abund = sum(1 for x in data if x <= 10)

    # Precompute F_i (frequency counts for i = 1 to 10)
    #Fi_abund = [sum(1 for x in data if x == i) for i in range(1, 11)]

    Fi_abund = []
    #for i in range(1, 11):
     #   Fi_abund[i] = sum(1 for x in data if x == i)
    for i in range(0, len(data)):
        summe = 1
        for j in range(0, len(data)):
            if i == j:
                continue
            if(data[i] == data[j]):
                summe = summe + 1

        #Fi_abund.append(summe)

    for i in range(1, 10):
        summe = 0
        for j in range(0, len(data)):
            if data[j] == i:

                summe = summe + 1
        Fi_abund.append(summe)
    # Calculate ACE and gamma²
    ace_result, gamma_sq_ace_result = ace(S_abund, S_rare_abund, F1_abund, N_rare_abund, Fi_abund)
    return ace_result, gamma_sq_ace_result


# ACE test
def test_ace():
    data = get_abundance_data()
    ace_result, gamma_sq_ace_result = analyze_data_ace(data)
    #print(f"ACE result: {ace_result}, Gamma²_ACE result: {gamma_sq_ace_result}")
    print(f"ACE result: {ace_result}")
    #print(gamma_sq_ace_result)


# ACE with modified Gamma² --> with the formula from the new paper

def calculate_C_ace(F1_abund, N_rare_abund):
    """
    Calculates sample coverage (C_ace).
    """
    if N_rare_abund == 0:
        return 0  # if N_rare == 0
        #return 1
    return 1 - (F1_abund / N_rare_abund)


def calculate_gamma_sq_ace(S_rare_abund, C_ace, Fi_abund, N_rare_abund):
    """
    Calculates gamma² for the standard ACE estimator.
    """
    if C_ace == 0 or N_rare_abund <= 1:
        return 0

    #sum_term = sum((i + 1) * i * Fi_abund[i] for i in range(min(10, len(Fi_abund))))
    sum_term = sum(i * (i - 1) * Fi_abund[i - 1] for i in range(1, min(11, len(Fi_abund) + 1)))
    gamma_sq_ace = max(0, (S_rare_abund / C_ace) * (sum_term / (N_rare_abund * (N_rare_abund - 1))) - 1)

    return gamma_sq_ace


def calculate_gamma_sq_ace_modified(S_ace, Fi_abund, N_rare_abund):
    """
    Calculates gamma² for the modified ACE estimator.
    """
    if N_rare_abund <= 1:
    #if N_rare_abund == 0:
        return 0
        #return 1

    #sum_term = sum((i + 1) * i * Fi_abund[i] for i in range(min(10, len(Fi_abund))))
    sum_term = sum(i * (i - 1) * Fi_abund[i - 1] for i in range(1, min(11, len(Fi_abund) + 1)))
    gamma_sq_ace_modified = max(0, S_ace * (sum_term / (N_rare_abund * (N_rare_abund - 1))) - 1)

    return gamma_sq_ace_modified


def ace_modified(S_abund, S_rare_abund, F1_abund, N_rare_abund, Fi_abund):
    """
    Calculates the ACE-modified estimator.
    """
    C_ace = calculate_C_ace(F1_abund, N_rare_abund)
    #print(f"C_ace: {C_ace}")

    # C_ace == 0
    if C_ace == 0:
        #print ("C_ace = 0")
        return S_abund + S_rare_abund, 0

    # first: calculate the regular ACE estimator
    gamma_sq_ace = calculate_gamma_sq_ace(S_rare_abund, C_ace, Fi_abund, N_rare_abund)
    S_ace = S_abund + (S_rare_abund / C_ace) + (F1_abund / C_ace) * gamma_sq_ace
    #print(f"Regular ACE (S_ace): {S_ace}")

    # second: use the result from S_ace to calculate the modified gamma_sq_ace
    gamma_sq_ace_modified = calculate_gamma_sq_ace_modified(S_ace, Fi_abund, N_rare_abund)
    #print(f"Modified gamma_sq_ace: {gamma_sq_ace_modified}")

    # calculate ACE-modified with modified gamma²
    S_ace_modified = S_abund + (S_rare_abund / C_ace) + (F1_abund / C_ace) * gamma_sq_ace_modified
    return S_ace_modified, gamma_sq_ace_modified


def analyze_data_ace_modified(data):
    """
    Processes data and calculates the ACE-modified estimator.
    """
    data.reverse()
    S_abund = sum(1 for x in data if x > 10)
    N_rare_abund = sum(x for x in data if x <= 10)
    F1_abund = sum(1 for x in data if x == 1)
    S_rare_abund = sum(1 for x in data if x <= 10)

    Fi_abund = [sum(1 for x in data if x == i) for i in range(1, 11)]

    #ace_result_modified, gamma_sq_ace_result_modified = ace_modified(S_abund, S_rare_abund, F1_abund, N_rare_abund, Fi_abund)
    #return ace_result_modified, gamma_sq_ace_result_modified
    return ace_modified(S_abund, S_rare_abund, F1_abund, N_rare_abund, Fi_abund)


# ACE modified test
def test_ace_modified():
    data = get_abundance_data()
    ace_result_modified, gamma_sq_ace_result_modified = analyze_data_ace_modified(data)
    print(f"Modified ACE result: {ace_result_modified}")
    #print(f"Gamma² Modified ACE result: {gamma_sq_ace_result_modified}")


# ICE Estimator
def calculate_C_ice(Q1, N_inf):
    """
    Calculates the sample incidence coverage estimator (C_ice).
    Args:
        Q1: Number of singletons (species observed in exactly one sample).
        N_inf: Total number of infrequent species.
    Returns:
        Coverage estimate for infrequent species.
    """
    """if N_inf == 0:
        return 0
    # division by zero
    return 1 - (Q1 / N_inf) """

    return 1 - (Q1 / N_inf) if N_inf > 0 else 0


def calculate_gamma_sq_ice(S_inf, C_ice, m_inf, N_inf, Qj):
    """
    Calculates the squared coefficient of variation (γ²_ice).

    Args:
        S_inf: Number of infrequent species.
        C_ice: Sample incidence coverage estimator.
        m_inf: Number of samples containing infrequent species.
        N_inf: Total count of infrequent species.
        Qj: List of frequencies for species occurring j times.
    Returns:
        Squared coefficient of variation (γ²_ice).
    """

    #if C_ice == 0 or N_inf < 1 or m_inf <= 1:
    if C_ice == 0 or N_inf <= 1 or m_inf <= 1:
        return 0

    # Sum from j=1 to 10 of j(j-1)Qj
    #sum_term = sum(j * (j - 1) * Qj[j - 1] for j in range(1, 10)) # MARTIN FRAGEN welche range eher die richtige ist
    sum_term = sum(j * (j - 1) * Qj[j - 1] for j in range(1, 11))

    # Calculate γ²_ice according to the formula
    """gamma_sq_ice = max(0, (S_inf / C_ice) * (m_inf / (m_inf - 1)) * (sum_term / (N_inf * N_inf)) - 1)"""
    #gamma_sq_ice = max((S_inf / C_ice) * (m_inf / (m_inf - 1)) * (sum_term / (N_inf * N_inf)) - 1, 0)
    gamma_sq_ice = max(0, (S_inf / C_ice) * (m_inf / (m_inf - 1)) * (sum_term / (N_inf * N_inf)) - 1)
    # DEBUG
    #print(f"calculate_gamma_sq_ice Debugging Info:\n"
          #f"S_inf: {S_inf}, C_ice: {C_ice}, m_inf: {m_inf}, N_inf: {N_inf}, Qj: {Qj}\n"
          #f"sum_term: {sum_term}, gamma_sq_ice: {gamma_sq_ice}")

    return gamma_sq_ice


def ice(S_freq, S_inf, Q1, N_inf, Qj, m_inf):
    """
    Calculates the ICE estimator for species richness.

    Args:
        S_freq: Number of frequent species.
        S_inf: Number of infrequent species.
        Q1: Number of singletons (species observed in exactly one sample).
        N_inf: Total count of infrequent species.
        Qj: List of frequencies for species occurring j times.
        m_inf: Number of samples containing infrequent species.
    Returns:
        Estimated species richness (S_ice) and γ²_ice.
    """
    C_ice = calculate_C_ice(Q1, N_inf)
    #print(f"C_ice: {C_ice}")
    gamma_sq_ice = calculate_gamma_sq_ice(S_inf, C_ice, m_inf, N_inf, Qj)
    #print(f"gamma_sq_ice: {gamma_sq_ice}")

    if C_ice == 0:
        return S_freq + S_inf, gamma_sq_ice

    S_ice = S_freq + (S_inf / C_ice) + (Q1 / C_ice) * gamma_sq_ice
    return S_ice, gamma_sq_ice


def analyze_ice_data(species_counts, data_log):
    species_counts = [int(x) for x in species_counts if str(x).isdigit()]

    S_freq = sum(1 for x in species_counts if x > 10)
    S_inf = sum(1 for x in species_counts if x <= 10)
    N_inf = sum(x for x in species_counts if x <= 10)
    Q1 = sum(1 for x in species_counts if x == 1)
    Qj = [sum(1 for x in species_counts if x == j) for j in range(1, 11)] # ?

    #print(f"S_freq: {S_freq}, S_inf: {S_inf}, N_inf: {N_inf}, Q1: {Q1}")

    # Calculate m_inf (number of samples with rare species)
    m_inf = sum(1 for trace in data_log if any(
        species_counts[ord(species) - ord('A')] <= 10
        for species in trace))

    # Calculate Qj (frequency of species that occur j times)
    """
    Qj = {j: 0 for j in range(1, 13)}
    for row in data_log:
        species_count = {}
        for species in row:
            species_count[species] = species_count.get(species, 0) + 1
        for count in species_count.values():
            if count <= 10:
                Qj[count] += 1
    # list of the Qj values
    Qj = [1, 1, 2, 1, 1, 1, 0, 0, 0, 0]

    Qj = [0] * 10
    for i in range(len(species_counts)):
        #if (species_counts[i]< 10):
        if (species_counts[i] <= 10):
            Qj[species_counts[i]-1] += 1

    Qj = [sum(1 for x in species_counts if x == j) for j in range(1, 11)]

    ice_result, gamma_sq_ice_result = ice(S_freq, S_inf, Q1, N_inf, Qj, m_inf)

    print(f"ICE estimator result: {ice_result}")
    #print(f"Gamma^2_ice result: {gamma_sq_ice_result}")

    return ice_result, gamma_sq_ice_result """

    #Qj = [sum(1 for x in species_counts if x == j) for j in range(1, 11)]

    return ice(S_freq, S_inf, Q1, N_inf, Qj, m_inf)


# ICE test
def test_ice():
    species_counts, data_log = get_species_counts_and_log()
    ice_result, gamma_sq_ice_result = analyze_ice_data(species_counts, data_log)
    #print(f"ICE result: {ice_result}, Gamma²_ICE result: {gamma_sq_ice_result}")
    print(f"ICE result: {ice_result}")


# ICE_modified Estimator
def calculate_C_ice(Q1, N_inf):
    if N_inf == 0:
        return 0
    return 1 - (Q1 / N_inf)


def calculate_gamma_sq_ice(S_inf, C_ice, m_inf, N_inf, Qj):
    if C_ice == 0 or N_inf < 1 or m_inf <= 1:
        return 0
    sum_term = sum(j * (j - 1) * Qj[j - 1] for j in range(1, 11))
    """gamma_sq_ice = max((S_inf / C_ice) * (m_inf / (m_inf - 1)) * (sum_term / (N_inf * (N_inf - 1))) - 1, 0)

    return gamma_sq_ice """
    gamma_sq_ice = max(
        (S_inf / C_ice) * (m_inf / (m_inf - 1)) * (sum_term / (N_inf * (N_inf - 1))) - 1, 0)
    return gamma_sq_ice


def calculate_gamma_sq_ice_modified(S_ice, m_inf, N_inf, Qj):
    if N_inf < 1 or m_inf <= 1:
        return 0

    sum_term = sum(j * (j - 1) * Qj[j - 1] for j in range(1, 11))
    # Modified formula uses S_ice directly instead of S_inf/C_ice
    gamma_sq_ice_modified = max(S_ice * (m_inf / (m_inf - 1)) * (sum_term / (N_inf * (N_inf - 1))) - 1, 0)
    return gamma_sq_ice_modified


def ice_modified(S_freq, S_inf, Q1, N_inf, Qj, m_inf):
    C_ice = calculate_C_ice(Q1, N_inf)
    if C_ice == 0 or C_ice == 1:
        return S_freq + S_inf, 0

    # First: Calculate the regular ICE
    gamma_sq_ice = calculate_gamma_sq_ice(S_inf, C_ice, m_inf, N_inf, Qj)
    S_ice = S_freq + (S_inf / C_ice) + (Q1 / C_ice) * gamma_sq_ice

    # Second: Use S_est to calculate the modified gamma_sq_ice
    gamma_sq_ice_modified = calculate_gamma_sq_ice_modified(S_ice, m_inf, N_inf, Qj)

    # Calculate the modified ICE estimator
    S_ice_modified = S_freq + (S_inf / C_ice) + (Q1 / C_ice) * gamma_sq_ice_modified
    return S_ice_modified, gamma_sq_ice_modified


def analyze_ice_data_modified(species_counts, data_log):
    if not species_counts or not data_log:
        raise ValueError("Empty species counts or data log provided")

    species_counts = [int(x) for x in species_counts if str(x).isdigit()]
    S_freq = sum(1 for x in species_counts if x > 10)
    S_inf = sum(1 for x in species_counts if x <= 10)
    N_inf = sum(x for x in species_counts if x <= 10)
    Q1 = sum(1 for x in species_counts if x == 1)

    # Calculate m_inf
    m_inf = sum(1 for trace in data_log if any(
        species_counts[ord(species) - ord('A')] <= 10
        for species in trace))

    # Calculate Qj
    Qj = [sum(1 for x in species_counts if x == j) for j in range(1, 11)]

    """"
    ice_result_modified, gamma_sq_ice_result_modified = ice_modified(S_freq, S_inf, Q1, N_inf, Qj, m_inf)
    return ice_result_modified, gamma_sq_ice_result_modified """
    return ice_modified(S_freq, S_inf, Q1, N_inf, Qj, m_inf)


# ICE modified test
def test_ice_modified():
    species_counts, data_log = get_species_counts_and_log()
    ice_result_modified, gamma_sq_ice_result_modified = analyze_ice_data_modified(species_counts, data_log)
    print(f"Modified ICE estimator result: {ice_result_modified}")


# Jackknife1 Estimator

def jackknife1_abundance(S_obs, f1):
    S_jackknife1 = S_obs + f1
    return S_jackknife1

# Jackknife1 test --> abundance based


def test_jackknife1_abundance():
    abundance_data = get_abundance_data()
    S_obs = len(abundance_data)  # S_obs
    f1 = sum(1 for x in abundance_data if x == 1)  # singletons
    jack1_abundance_result = jackknife1_abundance(S_obs, f1)

    #print("\nJackknife-1 Abundance-based Results:")
    #print(f"Observed Species (S_obs): {S_obs}")
    #print(f"Singletons (f1): {f1}")
    #print(f"Jackknife-1 Estimate: {jack1_result}")
    return jack1_abundance_result


def jackknife1_incidence(S_obs, Q1, m):
    if m == 0:
        return S_obs  # return observed species if no samples
        #return 0.0 # if m = 0 --> return 0
    return S_obs + Q1 * (m - 1) / m


def test_jackknife1_incidence():
    species_counts, data_log = get_species_counts_and_log()
    S_obs = len(species_counts)
    Q1 = species_counts.count(1)
    m = len(data_log)

    jack1_incidence_result = jackknife1_incidence(S_obs, Q1, m)

    #print("\nJackknife-1 Incidence-based Results:")
    #print(f"Observed Species (S_obs): {S_obs}")
    #print(f"Singletons (Q1): {Q1}")
    #print(f"Total Samples (m): {m}")
    #print(f"Jackknife-1 Estimate: {jack1_incidence_result}")
    return jack1_incidence_result


# Jackknife2 Estimator

def jackknife2_abundance(S_obs, f1, f2):
    S_jackknife2 = S_obs + 2 * f1 - f2
    return S_jackknife2


# Jackknife2 test --> abundance-based data

def test_jackknife2_abundance():
    abundance_data = get_abundance_data()
    S_obs = len(abundance_data)
    f1 = sum(1 for x in abundance_data if x == 1)
    f2 = sum(1 for x in abundance_data if x == 2)

    jack2_abundance_result = jackknife2_abundance(S_obs, f1, f2)

    #print("\nJackknife-2 Abundance-based Results:")
    #print(f"Observed Species (S_obs): {S_obs}")
    #print(f"Singletons (f1): {f1}")
    #print(f"Species observed twice (f2): {f2}")
    #print(f"Jackknife-2 Estimate: {jack2_abundance_result}")
    return jack2_abundance_result


# Test function (for incidence-based)
def jackknife2_incidence(S_obs, Q1, Q2, m):
    if m <= 1:
        return S_obs
    term1 = Q1 * (2 * m - 3) / m
    term2 = Q2 * ((m - 2) ** 2) / (m * (m - 1))
    return S_obs + term1 - term2


def test_jackknife2_incidence():
    species_counts, data_log = get_species_counts_and_log()
    S_obs = len(species_counts)
    Q1 = species_counts.count(1)
    Q2 = species_counts.count(2)

    m = len(data_log)
    jack2_incidence_result = jackknife2_incidence(S_obs, Q1, Q2, m)

    #print("\nJackknife-2 Incidence-based Results:")
    #print(f"Observed Species (S_obs): {S_obs}")
    #print(f"Singletons (Q1): {Q1}")
    #print(f"Species observed twice (Q2): {Q2}")
    #print(f"Total Samples (m): {m}")
    #print(f"Jackknife-2 Estimate: {jack2_incidence_result}")
    return jack2_incidence_result


# Chao1
def chao1(S_obs, f1, f2):
    """
    Args:
        S_obs (int): Number of observed species
        f1 (int): Number of species observed only once --> singletons
        f2 (int): Number of species observed exactly twice --> doubletons
    """
    if f2 == 0:
        result = S_obs + (f1 ** 2) / 2
    else:
        result = S_obs + (f1 ** 2) / (2 * f2)

    print(f"Chao1 result: {result}")
    return result


def test_chao1():
    abundance_data = get_abundance_data()
    S_obs = len(abundance_data)
    abundance_counts = Counter(abundance_data)
    f1 = abundance_counts[1]
    f2 = abundance_counts[2]

    result = chao1(S_obs, f1, f2)
    print(f"Chao1 result: {result}\n")


# Chao2
def chao2(S_obs, Q1, Q2):
    """
    Args:
        S_obs (int): Number of observed species
        Q1 (int): Number of species occurring in exactly one sample --> uniques
        Q2 (int): Number of species occurring in exactly two samples --> duplicates
    """
    if Q2 == 0:
        return S_obs + (Q1 ** 2) / 2
    return S_obs + (Q1 ** 2) / (2 * Q2)


def test_chao2():
    species_counts, data_log = get_species_counts_and_log()
    S_obs = len(set(species for sample in data_log for species in sample))

    species_occurrences = Counter(species for sample in data_log for species in set(sample))
    Q1 = sum(1 for count in species_occurrences.values() if count == 1)
    Q2 = sum(1 for count in species_occurrences.values() if count == 2)

    result = chao2(S_obs, Q1, Q2)
    print(f"Chao2 result: {result}\n")
    print(f"Observed Species (S_obs): {S_obs}")
    print(f"Singletons (Q1): {Q1}")
    print(f"Doubletons (Q2): {Q2}")
    print(f"Chao2 Estimate: {result}")
    return result


# iCHAO1 --> abundance based

def calculate_frequencies(traces):
    species_occurrences = defaultdict(int)

    for trace in traces:
        unique_species = set(trace)
        for species in unique_species:
            species_occurrences[species] += 1

    occurrence_counts = defaultdict(int)
    for count in species_occurrences.values():
        occurrence_counts[count] += 1

    #return species_occurrences

    f1 = occurrence_counts.get(1, 0)
    f2 = occurrence_counts.get(2, 0)
    f3 = occurrence_counts.get(3, 0)
    f4 = occurrence_counts.get(4, 0)

    return f1, f2, f3, f4


def chao1(S_obs, f1, f2):
    if f2 == 0:
        return S_obs + (f1 * (f1 - 1)) / 2
    return S_obs + (f1 ** 2) / (2 * f2)


def iChao1(S_chao1, f1, f2, f3, f4):
    if f4 == 0:
        if f3 == 0:
            return S_chao1
        correction_term = (f3 / 4) * max(f1 - (f2 * f3) / (2 * (f3 + 1)), 0)
    else:
        correction_term = (f3 / (4 * f4)) * max(f1 - (f2 * f3) / (2 * f4), 0)

    return S_chao1 + correction_term


# iChao1 test --> abundance-based data
def test_iChao1_abundance():
    data = get_abundance_data()
    S_obs = len(data)
    f1 = data.count(1)
    f2 = data.count(2)
    f3 = data.count(3)
    f4 = data.count(4)

    S_chao1 = chao1(S_obs, f1, f2)
    iChao1_abundance = iChao1(S_chao1, f1, f2, f3, f4)

    print(f"iChao1 result with abundance based data: {iChao1_abundance}")


# iCHAO2 --> incidence based
def calculate_Q_frequencies(traces):
    species_occurrences = defaultdict(int)

    for trace in traces:
        unique_species = set(trace)
        for species in unique_species:
            species_occurrences[species] += 1

    occurrence_counts = Counter(species_occurrences.values())

    Q1 = occurrence_counts.get(1, 0)
    Q2 = occurrence_counts.get(2, 0)
    Q3 = occurrence_counts.get(3, 0)
    Q4 = occurrence_counts.get(4, 0)

    return Q1, Q2, Q3, Q4


def chao2(S_obs, Q1, Q2):
    if Q2 == 0:
        return S_obs + (Q1 * (Q1 - 1)) / 2
    return S_obs + (Q1 * Q1) / (2 * Q2)


def iChao2(S_chao2, Q1, Q2, Q3, Q4, T):
    if Q4 == 0 or T <= 3:
        return S_chao2

    factor1 = (T - 3) / (4 * T)
    factor2 = Q3 / Q4
    nested_term = ((T - 3) / (2 * (T - 1))) * (Q2 * Q3 / Q4) if T > 1 else 0
    factor3 = max(Q1 - nested_term, 0)

    correction_term = factor1 * factor2 * factor3
    return S_chao2 + correction_term


# iChao2 test --> incidence-based data
def test_iChao2_incidence():
    species_counts, traces = get_species_counts_and_log()
    S_obs = len(set().union(*traces))
    Q1, Q2, Q3, Q4 = calculate_Q_frequencies(traces)
    T = len(traces)
    S_chao2 = chao2(S_obs, Q1, Q2)
    iChao2_incidence = iChao2(S_chao2, Q1, Q2, Q3, Q4, T)

    print(f"iChao2 result with incidence based data: {iChao2_incidence}")


# Run all tests
if __name__ == "__main__":
    test_ace()
    test_ace_modified()
    test_ice()
    test_ice_modified()
    test_jackknife1_incidence()
    test_jackknife1_abundance()
    test_jackknife2_incidence()
    test_jackknife2_abundance()
    test_chao1()
    test_chao2()
    test_iChao1_abundance()
    test_iChao2_incidence()

### Nina end


def estimate_exp_shannon_entropy_abundance(obs_species_counts: dict, sample_size: int) -> float:
    """
    computes the asymptotic(=estimated) exponential of Shannon entropy for abundance-based data
    :param obs_species_counts: the species with corresponding incidence counts
    :param sample_size: the sample size associated with the species incidence counts
    :return: the estimated exponential of Shannon entropy
    """
    if sum(obs_species_counts.values())==0 or sample_size==0:
        return 0.0
    return math.exp(estimate_entropy(obs_species_counts, sample_size))


def estimate_exp_shannon_entropy_incidence(obs_species_counts: dict, sample_size: int) -> float:
    """
    computes the asymptotic(=estimated) exponential of Shannon entropy for incidence-based data
    :param obs_species_counts: the species with corresponding incidence counts
    :param sample_size: the sample size associated with the species incidence counts
    :return: the estimated exponential of Shannon entropy
    """
    # term h_o is structurally equivalent to abundance based entropy estimation, see eq H7 in appendix H of Hill number paper
    u = sum(obs_species_counts.values())
    h_o = estimate_entropy(obs_species_counts, sample_size)
    if u == 0:
        return 0.0
    #print(h_o, u, sample_size)
    #print(">>>"+str(math.exp((sample_size / u) * h_o + math.log(u / sample_size))))
    return math.exp((sample_size / u) * h_o + math.log(u / sample_size))


def estimate_entropy(obs_species_counts: dict, sample_size: int) -> float:
    """
    computes the estimated Shannon entropy
    :param obs_species_counts: the species with corresponding incidence counts
    :param sample_size: the sample size associated with the species incidence counts
    :return: the estimated exponential of Shannon entropy
    """
    if sample_size <= 1:
        return 0

    f_1 = get_singletons(obs_species_counts)
    f_2 = get_doubletons(obs_species_counts)

    entropy_known_species = 0

    for x_i in obs_species_counts.values():
        if x_i <= sample_size - 1:
            norm_factor = x_i / sample_size

            #decompose sum(1/x_i,...,1/sample_size) to um(1/1,...,1/sample_size)-sum(1/1,...,1/x_i-1)
            entropy_known_species = entropy_known_species + norm_factor * (harmonic(sample_size) - harmonic(x_i-1))
            #entropy_known_species = entropy_known_species + norm_factor * (mpmath.harmonic(sample_size) - mpmath.harmonic(x_i-1))
            #entropy_known_species = entropy_known_species + norm_factor * sum([1 / k for k in range(x_i, sample_size)])

    #print(f_2,f_1, sample_size, sum(obs_species_counts.values()))
    a = 0
    if f_2 > 0:
        a = (2 * f_2) / ((sample_size - 1) * f_1 + 2 * f_2)
        #a = (2 * f_2) / ( f_1 + 2 * f_2)
    elif f_2 == 0 and f_1 > 0:
        a = 2 / ((sample_size - 1) * (f_1 - 1) + 2)
    else:
        a = 1
    #print(f_1, f_2, a, sample_size)
    entropy_unknown_species = 0
    if f_1==1 and f_2 >= 20:
        return entropy_known_species
    # TODO rethink if this is really necessary
    #if a == 1:
    #    return entropy_known_species
    #(((1 - a) ** (-sample_size + 1)))
    #print(0 ** 690)
    #print(((1 - a) ** (sample_size - 1)), sample_size, a)
    #print(entropy_known_species, a, f_1, f_2, sample_size)
    #print(a, f_1, f_2, sample_size)
    if a==1:
        return entropy_known_species
        #entropy_unknown_species = (f_1 / sample_size) * sum([(1 / r) for r in range(1, sample_size)])
        #return entropy_known_species + entropy_unknown_species
    ###((1-a) ** (-sample_size+1))
    #else:
    else:
        entropy_unknown_species = (f_1 / sample_size) * (1-a) ** (-sample_size+1) * (
                -math.log(a) - sum([1/r * ((1 - a) ** r) for r in range(1, sample_size)]))
        #print(entropy_known_species, entropy_unknown_species)
        #print(entropy_known_species, entropy_unknown_species, (f_1 / sample_size), ((1-a) ** (-sample_size+1)), -math.log(a), sum([(1 / r) * ((1 - a) ** r) for r in range(1, sample_size)]))
        #print("TEST "+str(entropy_known_species + entropy_unknown_species))
        #print(-math.log(a), sum([(1 / r) * ((1 - a) ** r) for r in range(1, sample_size)]), -math.log(a) - sum([(1 / r) * ((1 - a) ** r) for r in range(1, sample_size)]))
        return entropy_known_species + entropy_unknown_species


def harmonic(n):
    """Returns an (approximate) value of n-th harmonic number.
    If n>100, use an efficient approximation using the digamma function instead
    http://en.wikipedia.org/wiki/Harmonic_number
    taken from: https://stackoverflow.com/questions/404346/python-program-to-calculate-harmonic-series
     """
    if n <= 100:
        return sum(1/k for k in range(1,n+1))
    else:
        return digamma(n + 1) + euler_gamma


def estimate_simpson_diversity_abundance(obs_species_counts: dict, sample_size: int) -> float:
    """
    computes the asymptotic(=estimated) Simpson diversity for abundance-based data
    :param obs_species_counts: the species with corresponding incidence counts
    :param sample_size: the sample size associated with the species incidence counts
    :return: the estimated Simpson diversity
    """
    # TODO make this understandable
    denom = 0
    for x_i in obs_species_counts.values():
        if x_i >= 2:
            denom = denom + (x_i * (x_i - 1))
    if denom == 0:
        return 0
    return (sample_size * (sample_size - 1)) / denom


def estimate_simpson_diversity_incidence(obs_species_counts: dict, sample_size: int) -> float:
    """
    computes the asymptotic(=estimated) Simpson diversity for incidence-based data
    :param obs_species_counts: the species with corresponding incidence counts
    :param sample_size: the sample size associated with the species incidence counts
    :return: the estimated Simpson diversity
    """
    # TODO make this understandable
    u = get_total_species_count(obs_species_counts)
    s = 0
    if u == 0:
        return 0.0
    nom = ((1 - (1 / sample_size)) * u) ** 2

    for y_i in obs_species_counts.values():
        if y_i > 1:
            #    s = s + (sample_size ** 2 * y_i ** 2) / (u ** 2 * sample_size ** 2)
            s = s + (y_i * (y_i - 1))
    if s == 0:
        return 0
    # return s ** (1 / (1 - 2))
    return nom / s


def completeness(obs_species_counts: dict) -> float:
    """
    computes the completeness of the sample data. A value of '1' indicates full completeness,
    whereas as value of '0' indicates total incompleteness
    :param obs_species_counts: the species with corresponding incidence counts
    :return: the estimated completeness
    """
    obs_species_count = get_number_observed_species(obs_species_counts)
    s_P = estimate_species_richness_chao(obs_species_counts)
    if s_P == 0:
        return 0

    return obs_species_count / s_P


def coverage(obs_species_counts: dict, sample_size: int) -> float:
    """
    computes the coverage of the sample data. A value of '1' indicates full coverage,
    whereas as value of '0' indicates no coverage
    :param obs_species_counts: the species with corresponding incidence counts
    :param sample_size: the sample size associated with the species incidence counts
    :return: the estimated coverage
    """
    f_1 = get_singletons(obs_species_counts)
    f_2 = get_doubletons(obs_species_counts)
    Y = get_total_species_count(obs_species_counts)

    if sample_size == 0:
        return 0
    if f_2 == 0 and sample_size == 1:
        return 0
    if f_1 == 0 and f_2 == 0:
        return 1

    return 1 - f_1 / Y * (((sample_size - 1) * f_1) / ((sample_size - 1) * f_1 + 2 * f_2))


def sampling_effort_abundance(n: float, obs_species_counts: dict, sample_size: int) -> float:
    """
    computes the expected additional sampling effort needed to reach target completeness l for abundance data.
    If f exceeds the current completeness, this function returns 0
    :param n: desired target completeness
    :param obs_species_counts: the species with corresponding incidence counts
    :param sample_size: the sample size associated with the species incidence counts
    :return: the expected additional sampling effort
    """
    comp = completeness(obs_species_counts)
    f_1 = get_singletons(obs_species_counts)
    f_2 = get_doubletons(obs_species_counts)

    if n <= comp:
        return 0
    if f_2 == 0:
        return 0

    obs_species_count = estimate_species_richness_chao(obs_species_counts)
    #(get_number_observed_species(obs_species_counts))

    s_P = 0
    if f_2 != 0:
        s_P = f_1 ** 2 / (2 * f_2)
    else:
        s_P = f_1 * (f_1 - 1) / 2

    return ((sample_size * f_1) / (2 * f_2)) * math.log(s_P / ((1 - n) * (s_P + obs_species_count)))


def sampling_effort_incidence(n: float, obs_species_counts: dict, sample_size: int) -> float:
    """
    computes the expected additional sampling effort needed to reach target completeness l for incidence data.
    If f exceeds the current completeness, this function returns 0
    :param n: desired target completeness
    :param obs_species_counts: the species with corresponding incidence counts
    :param sample_size: the sample size associated with the species incidence counts
    :return: the expected additional sampling effort
    """
    comp = completeness(obs_species_counts)
    f_1 = get_singletons(obs_species_counts)
    f_2 = get_doubletons(obs_species_counts)
    if n <= comp:
        return 0
    #if f_2 == 0:
    #    return 0
    if sample_size==1:
        return 0
    if f_1 == 0:
        return 0

    # for small sample sizes, correction term is introduced, otherwise math error
    #obs_species_count = estimate_species_richness_chao(obs_species_counts)
    obs_species_count = get_number_observed_species(obs_species_counts)

    s_P = 0
    if f_2 != 0:
        s_P = obs_species_count + (1 - 1 / sample_size) * f_1 ** 2 / (2 * f_2)
    else:
        s_P = obs_species_count + (1 - 1 / sample_size) * f_1 * (f_1 - 1) / 2

    #s_P = obs_species_count + (1 - 1 / sample_size) * f_1 ** 2 / (2 * f_2)


    #TODO double check if this is indeed correct
    #should f_2 be 0, technically assessment is not possible, thus we treat it as if one doubletons remained.
    # Thus results are approximative in this case

    if f_2!=0:
        nom1 = (sample_size / (sample_size - 1))
        nom2 = ((2 * f_2) / (f_1 ** 2))
        nom3 = (n * s_P - obs_species_count)
        nom = (math.log(1 - nom1 * nom2 * nom3))
        denominator = (math.log(1 - ((2 * f_2) / ((sample_size - 1) * f_1 + 2 * f_2))))
        return nom / denominator
    else:
        nom1 = (sample_size / (sample_size - 1))
        nom2 = ((2 * 1) / (f_1 ** 2))
        nom3 = (n * s_P - obs_species_count)
        nom = (math.log(1 - nom1 * nom2 * nom3))
        denominator = (math.log(1 - ((2 * 1) / ((sample_size - 1) * f_1 + 2 * 1))))
        return nom / denominator
    #return final
