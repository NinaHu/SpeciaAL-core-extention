import math
from src.estimation.metrics import get_singletons


def extrapolate_richness_abundance(reference_sample, sample_size, richness, data_points=100):
    s_obs = len(reference_sample)
    f_0 = richness - s_obs
    f_1 = get_singletons(reference_sample)
    m_size = math.floor(sample_size / data_points)
    values = []
    locations = []
    m = 0
    while m + m_size < sample_size:
        m = m + m_size
        #print("Richness Extrapolation - M+m=" + str(sample_size + m))
        if f_0 == 0:
            values.append(s_obs)
        else:
            values.append(s_obs + f_0 * (1 - (1 - (f_1 / (sample_size * f_0 + f_1))) ** m))
        locations.append(sample_size + m)

    m =  sample_size
    if f_0 == 0:
        values.append(s_obs)
    else:
        values.append(s_obs + f_0 * (1 - (1 - (f_1 / (sample_size * f_0 + f_1))) ** m))
    locations.append(sample_size + m)
    return values, locations


# structurally equivalent to abundance case
def extrapolate_richness_incidence(reference_sample, sample_size, richness, data_points=100):
    return extrapolate_richness_abundance(reference_sample, sample_size, richness, data_points)


def extrapolate_shannon_entropy_abundance(reference_sample, sample_size, asymp_entr, data_points=100):
    m_size = math.floor(sample_size / data_points)
    asymp_entr = math.log(asymp_entr)
    values = []
    locations = []
    m = 0
    while m + m_size < sample_size:
        m = m + m_size
        #print("Shannon Entropy Extrapolation - M+m=" + str(sample_size + m))

        first = sample_size / (sample_size + m)
        second = (sum([- (x_i / sample_size) * math.log(x_i / sample_size) for x_i in reference_sample.values()]))
        third = m / (sample_size + m) * asymp_entr
        values.append(math.exp(first * second + third))
        locations.append(sample_size + m)

    m = sample_size
    first = sample_size / (sample_size + m)
    second = (sum([- (x_i / sample_size) * math.log(x_i / sample_size) for x_i in reference_sample.values()]))
    third = m / (sample_size + m) * asymp_entr
    values.append(math.exp(first * second + third))
    locations.append(sample_size + m)
    return values, locations


# structurally equivalent to abundance case with adapted sample size
def extrapolate_shannon_entropy_incidence(reference_sample, sample_size, asymp_entr, data_points=100):
    u = sum(list(reference_sample.values()))

    m_size = math.floor(sample_size / data_points)
    asymp_entr = math.log(asymp_entr)
    values = []
    locations = []
    m = 0
    while sample_size + m + m_size < 2 * sample_size:
        m = m + m_size
        #print("Shannon Entropy Extrapolation - M+m=" + str(sample_size + m))

        first = sample_size / (sample_size + m)
        second = (sum([- (y_i / u) * math.log(y_i / u) for y_i in reference_sample.values()]))
        third = m / (sample_size + m) * asymp_entr
        values.append(math.exp(first * second + third))
        locations.append(sample_size + m)

    m = 2 * sample_size
    first = sample_size / (sample_size + m)
    second = (sum([- (y_i / u) * math.log(y_i / u) for y_i in reference_sample.values()]))
    third = m / (sample_size + m) * asymp_entr
    values.append(math.exp(first * second + third))
    locations.append(m)
    return values, locations


def extrapolate_simpson_diversity_abundance(reference_sample, sample_size, data_points=100):
    m_size = math.floor(sample_size / data_points)

    values = []
    locations = []
    m = 0
    while sample_size + m + m_size < 2 * sample_size:
        m = m + m_size
        #print("Simpson Diversity Extrapolation - M+m=" + str(sample_size + m))

        first = 1 / (sample_size + m)
        second = (sample_size + m - 1) / (sample_size + m)
        third = sum([(x_i * (x_i - 1)) / (sample_size * (sample_size - 1)) for x_i in reference_sample.values()])
        values.append(1 / (first + second * third))
        locations.append(sample_size + m)

    m = 2 * sample_size
    first = 1 / (sample_size + m)
    second = (sample_size + m - 1) / (sample_size + m)
    third = sum([(x_i * (x_i - 1)) / (sample_size * (sample_size - 1)) for x_i in reference_sample.values()])
    values.append(1 / (first + second * third))
    locations.append(m)
    return values, locations


def extrapolate_simpson_diversity_incidence(reference_sample, sample_size, data_points=100):
    m_size = math.floor(sample_size / data_points)
    u = sum(list(reference_sample.values()))

    values = []
    locations = []
    m = 0
    while sample_size + m + m_size < 2 * sample_size:
        m = m + m_size
        #print("Simpson Diversity Extrapolation - M+m=" + str(sample_size + m))

        first = 1 / (sample_size + m)
        second = 1 / (u / sample_size)
        third = (sample_size + m - 1) / (sample_size + m)
        fourth = sum([(y_i * (y_i - 1)) / (u ** 2 * (1 - (1 / sample_size))) for y_i in reference_sample.values()])
        values.append(1 / ((first * second) + third * fourth))
        locations.append(sample_size + m)

    m = 2 * sample_size

    first = 1 / (sample_size + m)
    second = 1 / (u / sample_size)
    third = (sample_size + m - 1) / (sample_size + m)
    fourth = sum([(y_i * (y_i - 1)) / (u ** 2 * (1 - (1 / sample_size))) for y_i in reference_sample.values()])
    values.append(1 / ((first * second) + third * fourth))
    locations.append(m)
    return values, locations
