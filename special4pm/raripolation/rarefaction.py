import math
from functools import cache
from src.estimation.metrics import get_incidence_count
import gmpy2


def rarefy_richness_abundance(reference_sample, sample_size, goal, data_points=100):
    m_size = math.floor(sample_size / data_points)
    s_obs = len(reference_sample)
    values = []
    locations = []
    m = 0
    while m + m_size < sample_size:
        print("Richness Rarefaction - m=" + str(m))
        m = m + m_size
        s = 0
        for y_i in reference_sample.values():
            first = binom(sample_size - y_i, m)  # binom(sample_size - y_i, m)
            second = binom(sample_size, m)  # binom(sample_size, m)
            s = s + (first / second)
        values.append(s_obs - s)
        locations.append(m)
    values.insert(0, 0)
    locations.insert(0, 0)
    return values, locations


# structurally equal to abundance-based case
def rarefy_richness_incidence(reference_sample, sample_size, goal, data_points=100):
    return rarefy_richness_abundance(reference_sample, sample_size, goal, data_points)


def rarefy_shannon_entropy_abundance(reference_sample, sample_size, goal, data_points=100):
    incidences = []
    for x in set(reference_sample.values()):
        incidences.append((x, get_incidence_count(reference_sample, x)))
    incidences = tuple(incidences)

    # incidences[x] = get_incidence_count(reference_sample,x)
    m_size = math.floor(sample_size / data_points)
    values = []
    locations = []
    m = 0
    while m + m_size <= sample_size:
        print("Shannon Entropy Rarefaction - m=" + str(m))
        s = 0
        m = m + m_size

        for k in range(1, m + 1):
            first = (-(k / m) * math.log(k / m))
            second = get_expected_incidence_of_k_at_m(sample_size, k, m, incidences)
            v = first * second
            s = s + v
        values.append(math.exp(s))
        locations.append(m)
    values.insert(0, 0)
    locations.insert(0, 0)
    return values, locations


def rarefy_shannon_entropy_incidence(reference_sample, sample_size, goal, data_points=100):
    incidences = []
    for x in set(reference_sample.values()):
        incidences.append((x, get_incidence_count(reference_sample, x)))
    incidences = tuple(incidences)

    m_size = math.floor(sample_size / data_points)
    u = sum(list(reference_sample.values()))
    values = []
    locations = []
    m = 0
    while m + m_size <= sample_size:
        s = 0
        m = m + m_size
        print("Shannon Entropy Rarefaction - m=" + str(m))
        u_t = (m * u) / sample_size
        for k in range(1, m + 1):
            first = (-(k / u_t) * math.log(k / u_t))
            second = get_expected_incidence_of_k_at_m(sample_size, k, m, incidences)
            v = first * second
            # print(first, second)
            s = s + v
        values.append(math.exp(s))
        locations.append(m)
    values.insert(0, 0)
    locations.insert(0, 0)
    return values, locations


def rarefy_simpson_diversity_abundance(reference_sample, sample_size, goal, data_points=100):
    incidences = []
    for x in set(reference_sample.values()):
        incidences.append((x, get_incidence_count(reference_sample, x)))
    incidences = tuple(incidences)

    m_size = math.floor(sample_size / data_points)
    # s_obs = len(reference_sample)
    # u = sum(list(reference_sample.values()))
    values = []
    locations = []
    m = 0
    while m + m_size < sample_size:
        print("Simpson Diversity Rarefaction - m=" + str(m))
        m = m + m_size
        # u_t = m * u / sample_size
        value = 1 / (
            sum([(k / m) ** 2 * get_expected_incidence_of_k_at_m(sample_size, k, m, incidences) for k in
                 range(1, m + 1)]))
        values.append(value)
        locations.append(m)
    values.insert(0, 0)
    locations.insert(0, 0)
    return values, locations


def rarefy_simpson_diversity_incidence(reference_sample, sample_size, goal, data_points=100):
    incidences = []
    for x in set(reference_sample.values()):
        incidences.append((x, get_incidence_count(reference_sample, x)))
    incidences = tuple(incidences)
    m_size = math.floor(sample_size / data_points)
    u = sum(list(reference_sample.values()))
    values = []
    locations = []
    m = 0
    while m + m_size < sample_size:
        print("Simpson Diversity Rarefaction - m=" + str(m))
        m = m + m_size
        u_t = m * u / sample_size
        value = 1 / (
            sum([(k / u_t) ** 2 * get_expected_incidence_of_k_at_m(sample_size, k, m, incidences) for k in
                 range(1, m + 1)]))
        values.append(value)
        locations.append(m)
    values.insert(0, 0)
    locations.insert(0, 0)
    return values, locations


@cache
def get_expected_incidence_of_k_at_m(sample_size, k, m, incidences):
    s = 0
    # for j in set(reference_sample.values()):
    for j, f_j in incidences:
        # for j in range(k, sample_size):
        if j < k:
            continue
        # f_j = get_incidence_count(reference_sample, j)
        # if sample_size-j < m-k:
        #   continue
        # if f_j == 0:# or f_j > sample_size - m+k+1:
        #     continue

        # first = scipy.special4pm.comb(j, k, exact=True) if k <= j / 2 else scipy.special4pm.comb(j, j - k, exact=True)
        # second = scipy.special4pm.comb(sample_size - j, m - k, exact=True) if m - k <= (
        #        sample_size - j) / 2 else scipy.special4pm.comb(sample_size - j, sample_size - j - m - k, exact=True)
        # third = scipy.special4pm.comb(sample_size, m, exact=True) if m <= sample_size else scipy.special4pm.comb(sample_size,
        #                                                                                                   sample_size - m,
        #                                                                                                   exact=True)
        first = binom(j, k)  # if k <= j / 2 else binom(j, j - k)
        second = binom(sample_size - j,
                       m - k)  # if m - k <= (sample_size - j) / 2 else binom(sample_size - j, sample_size - j - m - k)
        third = binom(sample_size, m)  # if m <= sample_size else binom(sample_size,sample_size - m)
        # print(first, second, third)
        # s = s * f_j * math.exp(first + second - third)
        s = s + ((first * second) / third) * f_j
    return s

    # first, second, third = 0, 0, 0
    # s = 0
    # third = scipy.special4pm.comb(sample_size, m, exact=True) if m<=sample_size else scipy.special4pm.comb(sample_size, sample_size - m, exact=True)
    # # print()
    # for i, y_i in enumerate(reference_sample.values()):
    # #    # print(y_i)
    #     if y_i >= k:
    #         first = scipy.special4pm.comb(y_i,k, exact=True) if k<=y_i/2 else scipy.special4pm.comb(y_i,y_i-k, exact=True)
    #         if sample_size - y_i >= m - k:
    #             second = scipy.special4pm.comb(sample_size-y_i, m-k, exact=True) if m-k<=(sample_size-y_i)/2 else scipy.special4pm.comb(sample_size-y_i, sample_size-y_i - m-k, exact=True)
    # #            # print(first, second, first * second, third)
    # #    # print(first * second, third)
    #     s = s + (first * second) / (third)
    #
    # return s


@cache
def binom(n, k):
    # if k>n:
    #    return 0
    # return math.comb(n, k)
    # return scipy.special4pm.comb(n, k, exact=True)
    return gmpy2.comb(n, k)
    # """
    # A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    # """
    # if 0 <= k <= n:
    #     ntok = 1
    #     ktok = 1
    #     for t in range(1, min(k, n - k) + 1):
    #         ntok *= n
    #         ktok *= t
    #         n -= 1
    #     return ntok // ktok
    # else:
    #     return 0
    # """Computes n! / (r! (n-r)!) exactly. Returns a python long int."""
    # assert n >= 0
    # assert 0 <= r <= n
    #
    # c = 1
    # denom = 1
    # for (num, denom) in zip(range(n, n - r, -1), range(1, r + 1, 1)):
    #     c = (c * num) // denom
    # return c


def binomial(n, k):
    if k > n:
        return 0
    if k > n / 2:
        k = n - k
    prod = 1
    for i in range(1, k + 1):
        prod = prod * (n + 1 - i) / i
    # print("!!!!------------!!!!")
    return prod
    # return reduce(operator.mul, [(n+1-i)/i for i in range(1,k+1)], 1)

    # (math.prod([(n+1-i)/i for i in range(1,k+1)]))
