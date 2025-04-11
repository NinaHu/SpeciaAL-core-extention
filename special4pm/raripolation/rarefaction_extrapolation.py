import math
import statistics
import time


from src.bootstrap.bootstrap import generate_bootstrap_sequence_abundance, generate_bootstrap_sequence_incidence
from src.raripolation.extrapolation import extrapolate_richness_abundance, extrapolate_shannon_entropy_abundance, \
    extrapolate_simpson_diversity_abundance, extrapolate_richness_incidence, extrapolate_shannon_entropy_incidence, \
    extrapolate_simpson_diversity_incidence
from src.estimation.metrics import get_number_observed_species, entropy_exp, simpson_diversity, estimate_species_richness_chao, estimate_shannon_entropy_abundance, \
    estimate_shannon_entropy_incidence
from rarefaction import rarefy_richness_abundance, rarefy_shannon_entropy_abundance, rarefy_simpson_diversity_abundance, \
    rarefy_richness_incidence, rarefy_shannon_entropy_incidence, rarefy_simpson_diversity_incidence


def rarefy_extrapolate_q0(est, abundance=True, data_points=30):
    start = time.time()
    if abundance:
        print("Richness Abundance(" + str(est.observed_species_count[-1])+", "+ str(est.chao2_abundance[-1]) + ")")

        r, loc_r = rarefy_richness_abundance(est.reference_sample_abundance, est.number_observations_abundance,
                                             est.observed_species_count[-1], data_points)
        e, loc_e = extrapolate_richness_abundance(est.reference_sample_abundance, est.number_observations_abundance,
                                                  est.chao2_abundance[-1], data_points)
        print("Eplapsed Time: "+str(time.time() - start))
        return r.append + [est.observed_species_count[-1]] + e, loc_r + [est.number_observations_abundance] + loc_e
    else:
        print("Richness Incidence(" + str(est.observed_species_count[-1])+", "+ str(est.chao2_incidence[-1]) + ")")

        r, loc_r = rarefy_richness_incidence(est.reference_sample_incidence, est.number_observations_incidence,
                                             est.observed_species_count[-1], data_points)
        e, loc_e = extrapolate_richness_incidence(est.reference_sample_incidence, est.number_observations_incidence,
                                                  est.chao2_abundance[-1], data_points)
        print("Eplapsed Time: "+str(time.time() - start))
        return r + [est.observed_species_count[-1]] + e, loc_r + [est.number_observations_incidence] + loc_e


def rarefy_extrapolate_q1(est, abundance=True, data_points=30):
    start = time.time()
    if abundance:
        print("Shannon Abundance(" + str(est.D1_sample_abundance[-1])+", "+ str(est.D1_estimated_abundance[-1]) + ")")

        r, loc_r = rarefy_shannon_entropy_abundance(est.reference_sample_abundance, est.number_observations_abundance,
                                                    est.D1_sample_abundance[-1], data_points)
        e, loc_e = extrapolate_shannon_entropy_abundance(est.reference_sample_abundance,
                                                         est.number_observations_abundance,
                                                         est.D1_estimated_abundance[-1], data_points)
        print("Eplapsed Time: "+str(time.time() - start))
        return r + [est.D1_sample_abundance[-1]] + e, loc_r + [est.number_observations_abundance] + loc_e
    else:
        print("Shannon Incidence(" + str(est.D1_sample_incidence[-1])+", "+ str(est.D1_estimated_incidence[-1]) + ")")

        r, loc_r = rarefy_shannon_entropy_incidence(est.reference_sample_incidence, est.number_observations_incidence,
                                                    est.D1_sample_incidence[-1], data_points)
        e, loc_e = extrapolate_shannon_entropy_incidence(est.reference_sample_incidence,
                                                         est.number_observations_incidence,
                                                         est.D1_estimated_incidence[-1], data_points)
        print("Eplapsed Time: "+str(time.time() - start))
        return r + [est.D1_sample_incidence[-1]] + e, loc_r + [est.number_observations_incidence] + loc_e


def rarefy_extrapolate_q2(est, abundance=True, data_points=30):
    start = time.time()
    if abundance:
        print("Simpson Abundance(" + str(est.D2_sample_abundance[-1])+", "+ str(est.D2_estimated_abundance[-1]) + ")")

        r, loc_r = rarefy_simpson_diversity_abundance(est.reference_sample_abundance, est.number_observations_abundance,
                                                      est.D2_sample_abundance[-1], data_points)
        e, loc_e = extrapolate_simpson_diversity_abundance(est.reference_sample_abundance,
                                                           est.number_observations_abundance,
                                                           data_points)
        print("Eplapsed Time: "+str(time.time() - start))
        return r + [est.D2_sample_abundance[-1]] + e, loc_r + [est.number_observations_abundance] + loc_e
    else:
        print("Simpson Incidence(" + str(est.D2_sample_incidence[-1])+", "+ str(est.D2_estimated_incidence[-1]) + ")")
        r, loc_r = rarefy_simpson_diversity_incidence(est.reference_sample_incidence, est.number_observations_incidence,
                                                      est.D2_sample_incidence[-1], data_points)
        e, loc_e = extrapolate_simpson_diversity_incidence(est.reference_sample_incidence,
                                                           est.number_observations_incidence,
                                                           data_points)
        print("Eplapsed Time: "+str(time.time() - start))
        return r + [est.D2_sample_incidence[-1]] + e, loc_r + [est.number_observations_incidence] + loc_e


def rarefy_extrapolate_all(est, abundance_data=True, data_points=30):
    if abundance_data:
        q0_a, q0_a_loc = rarefy_extrapolate_q0(est, abundance=True, data_points=data_points)
        q1_a, q1_a_loc = rarefy_extrapolate_q1(est, abundance=True, data_points=data_points)
        q2_a, q2_a_loc = rarefy_extrapolate_q2(est, abundance=True, data_points=data_points)
        return (q0_a, q0_a_loc), (q1_a, q1_a_loc), (q2_a, q2_a_loc)
    else:
        q0_i, q0_i_loc = rarefy_extrapolate_q0(est, abundance=False, data_points=data_points)
        q1_i, q1_i_loc = rarefy_extrapolate_q1(est, abundance=False, data_points=data_points)
        q2_i, q2_i_loc = rarefy_extrapolate_q2(est, abundance=False, data_points=data_points)
        return (q0_i, q0_i_loc), (q1_i, q1_i_loc), (q2_i, q2_i_loc)



def rarefy_extrapolate_bootstrap_all(s, abundance_data=True, data_points=30, bootstrap_repetitions = 200):
    q0_all=[]
    q1_all=[]
    q2_all=[]

    q_loc=None
    print("Creatign Bootstrap Samples")
    for i in range(0, bootstrap_repetitions):
        #function results for all samples
        q0 = [0]
        q1 = [0]
        q2 = [0]

        seq_a = None
        loc_a = None
        #subsampling of bootstrap samples
        if abundance_data:
            seq_a, loc_a = generate_bootstrap_sequence_abundance(s.reference_sample_abundance,
                                                                 s.number_observations_abundance, math.floor(
                    s.number_observations_abundance / data_points))
        else:
            #print("OBSERVATIONS: "+str(s.number_observations_incidence))
            seq_a, loc_a = generate_bootstrap_sequence_incidence(s.reference_sample_incidence,
                                                                 s.number_observations_incidence, math.floor(
                    s.number_observations_incidence / data_points))


        for seq in seq_a:
            q0.append(get_number_observed_species(seq))
            q1.append(entropy_exp(seq,sum(seq.values())))
            q2.append(simpson_diversity(seq,sum(seq.values())))

        #extrapolation
        ref_sample = seq_a[-1]
        q0_e = None
        q1_e = None
        q2_e = None
        loc_e = None

        if abundance_data:
            q0_e, loc_e = extrapolate_richness_abundance(ref_sample, s.number_observations_abundance, estimate_species_richness_chao(ref_sample), data_points=data_points)
            q1_e, _ = extrapolate_shannon_entropy_abundance(ref_sample, s.number_observations_abundance,
                                                            math.exp(estimate_shannon_entropy_abundance(ref_sample, s.number_observations_abundance)), data_points= data_points)
            q2_e, _ = extrapolate_simpson_diversity_abundance(ref_sample, s.number_observations_abundance,data_points= data_points)
        else:
            q0_e, loc_e = extrapolate_richness_incidence(ref_sample, s.number_observations_incidence, estimate_species_richness_chao(ref_sample),
                                                         data_points=data_points)
            q1_e, _ = extrapolate_shannon_entropy_incidence(ref_sample, s.number_observations_incidence,
                                                            math.exp(estimate_shannon_entropy_incidence(ref_sample,
                                                                                                        s.number_observations_incidence)),
                                                            data_points=data_points)
            q2_e, _ = extrapolate_simpson_diversity_incidence(ref_sample, s.number_observations_incidence,
                                                              data_points=data_points)

        #connect subsampled and extrapolated function curves
        q0 = q0 + q0_e
        q1 = q1 + q1_e
        q2 = q2 + q2_e
        q_loc = loc_a + loc_e

        #add bootstrap function values to list of all results
        for x in range(0,len(q_loc)):
            if len(q0_all)<=x:
                q0_all.append([])
                q1_all.append([])
                q2_all.append([])
            q0_all[x].append(q0[x])
            q1_all[x].append(q1[x])
            q2_all[x].append(q2[x])

    #aggregate bootstrap results
    q0_mean = []
    q0_ciu = []
    q0_cil = []

    q1_mean = []
    q1_ciu = []
    q1_cil = []

    q2_mean = []
    q2_ciu = []
    q2_cil = []
    for x in range(0,len(q0_all)):
        q0_mean.append(statistics.mean(q0_all[x]))
        q0_ciu.append(q0_mean[x]+1.96*statistics.stdev(q0_all[x]))
        q0_cil.append(q0_mean[x]-1.96*statistics.stdev(q0_all[x]))

        q1_mean.append(statistics.mean(q1_all[x]))
        q1_ciu.append(q1_mean[x]+1.96*statistics.stdev(q1_all[x]))
        q1_cil.append(q1_mean[x]-1.96*statistics.stdev(q1_all[x]))

        q2_mean.append(statistics.mean(q2_all[x]))
        q2_ciu.append(q2_mean[x]+1.96*statistics.stdev(q2_all[x]))
        q2_cil.append(q2_mean[x]-1.96*statistics.stdev(q2_all[x]))

    return(q0_mean,q1_mean,q2_mean),(q0_cil,q1_cil,q2_cil),(q0_ciu,q1_ciu,q2_ciu), q_loc