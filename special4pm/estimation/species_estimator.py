from typing import Callable

import pandas as pd
import pm4py

from deprecation import deprecated
from pandas import DataFrame
from pm4py.objects.log.obj import EventLog, Trace
from special4pm.bootstrap import bootstrap
from tqdm import tqdm
from collections import Counter

from special4pm.estimation.metrics import get_singletons, get_doubletons, completeness, coverage, \
    sampling_effort_abundance, sampling_effort_incidence, hill_number_asymptotic, entropy_exp, simpson_diversity, ace, \
    ace_modified, ice, ice_modified, jackknife1_abundance, jackknife1_incidence, jackknife2_abundance, \
    jackknife2_incidence, chao1, chao2, iChao1, iChao2, calculate_C_ice #calculate_gamma_sq_ice, #calculate_Q_frequencies


# TODO enum for proper key access
# TODO redo print to be r-like table of current values or history of values
# TODO differentiate between abundance and incidence
# TODO maybe remove MetricManager and have class extend hash map directly
# TODO move bootstrap out of here
# TODO dataFrame incorporate all information

class MetricManager(dict):
    # TODO convert to dataclass
    """
    Manages metrics for abundance and incidence models.
    """
    def __init__(self, d0: bool, d1: bool, d2: bool, c0: bool, c1: bool, l_n: list, ace: bool = True,
                 ace_modified: bool = True, ice: bool = True, ice_modified: bool = True,
                 jackknife1_abundance: bool = True, jackknife1_incidence: bool = True, jackknife2_abundance: bool = True,
                 jackknife2_incidence: bool = True, chao1: bool = True, chao2: bool = True, iChao1: bool = True,
                 iChao2: bool = True) -> None:

        # reference sample stats
        super().__init__()
        self.reference_sample_abundance = {}
        self.reference_sample_incidence = {}

        self.incidence_current_total_species_count = 0
        self.abundance_current_total_species_count = 0
        self.incidence_sample_size = 0
        self.abundance_sample_size = 0
        self.current_co_occurrence = 0
        self.empty_traces = 0

        self["abundance_no_observations"] = [0]
        self["incidence_no_observations"] = [0]
        self["abundance_sum_species_counts"] = [0]
        self["incidence_sum_species_counts"] = [0]
        self["degree_of_co_occurrence"] = [0]
        self["abundance_singletons"] = [0]
        self["incidence_singletons"] = [0]
        self["abundance_doubletons"] = [0]
        self["incidence_doubletons"] = [0]
        if d0:
            self["abundance_sample_d0"] = [0]
            self["incidence_sample_d0"] = [0]
            self["abundance_estimate_d0"] = [0]
            self["incidence_estimate_d0"] = [0]
            self["incidence_estimate_d0_ci"] = [-1]

        if d1:
            self["abundance_sample_d1"] = [0]
            self["incidence_sample_d1"] = [0]
            self["abundance_estimate_d1"] = [0]
            self["incidence_estimate_d1"] = [0]
            self["incidence_estimate_d1_ci"] = [-1]

        if d2:
            self["abundance_sample_d2"] = [0]
            self["incidence_sample_d2"] = [0]
            self["abundance_estimate_d2"] = [0]
            self["incidence_estimate_d2"] = [0]
            self["incidence_estimate_d2_ci"] = [-1]

        if c0:
            self["abundance_c0"] = [0]
            self["incidence_c0"] = [0]
            self["incidence_c0_ci"] = [-1]

        if c1:
            self["abundance_c1"] = [0]
            self["incidence_c1"] = [0]
            self["incidence_c1_ci"] = [-1]

        for l in l_n:
            self["abundance_l_" + str(l)] = [0]
            self["incidence_l_" + str(l)] = [0]
            self["incidence_l_" + str(l)+"_ci"] = [-1]

        if ace:
            self["ace"] = [0]

        if ace_modified:
            self["ace_modified"] = [0]

        if ice:
            self["ice"] = [0]

        if ice_modified:
            self["ice_modified"] = [0]

        if jackknife1_abundance:
            self["jackknife1_abundance"] = [0]

        if jackknife1_incidence:
            self["jackknife1_incidence"] = [0]

        if jackknife2_abundance:
            self["jackknife2_abundance"] = [0]

        if jackknife2_incidence:
            self["jackknife2_incidence"] = [0]

        if chao1:
            self["chao1"] = [0]

        if chao2:
            self["chao2"] = [0]

        if iChao1:
            self["iChao1"] = [0]

        if iChao2:
            self["iChao2"] = [0]


class SpeciesEstimator:
    """
    A class for the estimation of diversity and completeness profiles of trace-based species definitions
    """

    def __init__(self, d0: bool = True, d1: bool = True, d2: bool = True, c0: bool = True,
                 c1: bool = True, ace: bool = True, ace_modified: bool = True, ice: bool = True, ice_modified: bool = True,
                 jackknife1_abundance: bool = True, jackknife1_incidence: bool = True, jackknife2_abundance: bool = True,
                 jackknife2_incidence: bool = True, chao1: bool = True, chao2: bool = True, iChao1: bool = True,
                 iChao2: bool = True, l_n: list = [.9, .95, .99], no_bootstrap_samples: int = 0, step_size: int | None = None):
        """
        :param d0: flag indicating if D0(=species richness) should be included
        :param d1: flag indicating if D1(=exponential Shannon entropy) should be included
        :param d2: flag indicating if D2(=Simpson diversity index) should be included
        :param c0: flag indicating if C0(=completeness) should be included
        :param c1: flag indicating if C1(=coverage) should be included
        :param l_n: list of desired completeness values for estimation additional sampling effort
        :param step_size: the number of added traces after which the profiles are updated. Use None if
        """
        # TODO add differentiation between abundance and incidence based data
        self.include_abundance = True
        self.include_incidence = True

        self.include_d0 = d0
        self.include_d1 = d1
        self.include_d2 = d2

        self.include_c0 = c0
        self.include_c1 = c1

        self.include_ace = ace
        self.include_ace_modified = ace_modified
        self.include_ice = ice
        self.include_ice_modified = ice_modified
        self.include_jackknife1_abundance = jackknife1_abundance
        self.include_jackknife1_incidence = jackknife2_incidence
        self.include_jackknife2_abundance = jackknife2_abundance
        self.include_jackknife2_incidence = jackknife2_incidence
        self.include_chao1 = chao1
        self.include_chao2 = chao2
        self.include_iChao1 = iChao1
        self.include_iChao2 = iChao2

        self.no_bootstrap_samples = no_bootstrap_samples

        self.l_n = l_n

        self.step_size = step_size

        self.metrics = {}
        self.species_retrieval = {}

        self.current_obs_empty = False

    def register(self, species_id: str, function: Callable) -> None:
        self.species_retrieval[species_id] = function
        self.metrics[species_id] = MetricManager(self.include_d0, self.include_d1, self.include_d2, self.include_c0,
                                                 self.include_c1, self.l_n, self.include_ace, self.include_ace_modified,
                                                 self.include_ice, self.include_ice_modified,
                                                 self.include_jackknife1_abundance, self.include_jackknife1_incidence,
                                                 self.include_jackknife2_abundance, self.include_jackknife2_incidence,
                                                 self.include_chao1, self.include_chao2,
                                                 self.include_iChao1, self.include_iChao2)

    def add_bootstrap_ci(self, sample_size):
        #print("Adding Bootstrapping Confidence Intervals")
        for species_id in self.metrics.keys():
            ci=(bootstrap.get_bootstrap_ci_incidence(self.metrics[species_id].reference_sample_incidence,
                                                       self.metrics[species_id].incidence_sample_size -
                                                       self.metrics[species_id].empty_traces, sample_size))

            self.metrics[species_id]["incidence_estimate_d0_ci"][-1] = ci[0]
            self.metrics[species_id]["incidence_estimate_d1_ci"][-1] = ci[1]
            self.metrics[species_id]["incidence_estimate_d2_ci"][-1] = ci[2]
            self.metrics[species_id]["incidence_c0_ci"][-1] = ci[3]
            self.metrics[species_id]["incidence_c1_ci"][-1] = ci[4]

        #    print(species_id + "Estimates + CI")
            d0 = self.metrics[species_id]["incidence_estimate_d0"][-1]
        #    print("D0: " + str(d0) + " (" + str(d0 - ci[0]), str(d0 + ci[0]) + ")")
            d1 = self.metrics[species_id]["incidence_estimate_d1"][-1]
        #    print("D1: " + str(d1) + " (" + str(d1 - ci[1]), str(d1 + ci[1]) + ")")
            d2 = self.metrics[species_id]["incidence_estimate_d2"][-1]
        #    print("D2: " + str(d2) + " (" + str(d2 - ci[2]), str(d2 + ci[2]) + ")")
            c0 = self.metrics[species_id]["incidence_c0"][-1]
        #    print("C0: " + str(c0) + " (" + str(c0 - ci[3]), str(c0 + ci[3]) + ")")
            c1 = self.metrics[species_id]["incidence_c1"][-1]
        #    print("C0: " + str(c1) + " (" + str(c1 - ci[4]), str(c1 + ci[4]) + ")")
        #    print()
        return

    def apply(self, data: pd.DataFrame | EventLog | Trace, verbose=True) -> None:
        """
        add all observations of an event log and update diversity and completeness profiles once afterward.
        If parameter step_size is set to an int, profiles are additionally updated along the way according to
        the step size
        :param data: the event log containing the trace observations
        """
        if isinstance(data, pd.DataFrame):
            return self.apply(pm4py.convert_to_event_log(data))


        #todo find out why this is notably faster than self.apply(tr) for tr in Log
        elif isinstance(data, EventLog) or isinstance(data, list):
            for species_id in self.species_retrieval.keys():
                #TODO find better way to
                for tr in tqdm(data, "Profiling Log for " + species_id, disable=not verbose):
                    self.add_observation(tr, species_id)
                    # if step size is set, update metrics after <step_size> many traces
                    if self.step_size is None:
                        continue
                    elif self.metrics[species_id].incidence_sample_size % self.step_size == 0:
                        self.update_metrics(species_id)
                if self.step_size is None or len(data) % self.step_size != 0:
                    self.update_metrics(species_id)
            if self.no_bootstrap_samples > 0:
                self.add_bootstrap_ci(self.no_bootstrap_samples)
        elif isinstance(data, Trace):
            for species_id in self.species_retrieval.keys():
                self.add_observation(data, species_id)
                # if step size is set, update metrics after <step_size> many traces
                if self.step_size is None:
                    continue
                elif self.metrics[species_id].incidence_sample_size % self.step_size == 0:
                    self.update_metrics(species_id)

        else:
            raise RuntimeError('Cannot apply data of type ' + str(type(data)))

    def add_observation(self, observation: Trace, species_id: str) -> None:
        """
        adds a single observation
        :param species_id: the species definition for which observation shall be added
        :param observation: the trace observation
        """
        # retrieve species from current observation
        species_abundance = self.species_retrieval[species_id](observation)
        species_incidence = set(species_abundance)
        if len(species_abundance) == 0:
            self.metrics[species_id].empty_traces = self.metrics[species_id].empty_traces + 1
            self.current_obs_empty = True
        else:
            self.current_obs_empty = False

        self.metrics[species_id].trace_retrieved_species_abundance = species_abundance
        self.metrics[species_id].trace_retrieved_species_incidence = species_incidence

        # update species abundances/incidences
        for s in species_abundance:
            self.metrics[species_id].reference_sample_abundance[s] = self.metrics[
                                                                         species_id].reference_sample_abundance.get(s,
                                                                                                                    0)+1

        for s in species_incidence:
            self.metrics[species_id].reference_sample_incidence[s] = self.metrics[
                                                                         species_id].reference_sample_incidence.get(s,
                                                                                                                    0)+1

        # update current number of observation for each model
        self.metrics[species_id].abundance_sample_size = self.metrics[species_id].abundance_sample_size + len(
            species_abundance)
        self.metrics[species_id].incidence_sample_size = self.metrics[species_id].incidence_sample_size + 1

        # update current sum of all observed species for each model
        self.metrics[species_id].abundance_current_total_species_count = \
            self.metrics[species_id].abundance_current_total_species_count + len(species_abundance)
        self.metrics[species_id].incidence_current_total_species_count = \
            self.metrics[species_id].incidence_current_total_species_count + len(
                species_incidence)

        # update current degree of spatial aggregation
        if self.metrics[species_id].incidence_current_total_species_count == 0:
            self.metrics[species_id].current_co_occurrence = 0
        else:
            self.metrics[species_id].current_co_occurrence = 1 - (
                    self.metrics[species_id].incidence_current_total_species_count / self.metrics[
                species_id].abundance_current_total_species_count)

    def update_metrics(self, species_id: str) -> None:
        """
        updates the diversity and completeness profiles based on the current observations
        """
        #  if self.current_obs_empty:
        #    return

        # update number of observations so far
        self.metrics[species_id]["abundance_no_observations"].append(self.metrics[species_id].abundance_sample_size)
        self.metrics[species_id]["incidence_no_observations"].append(self.metrics[species_id].incidence_sample_size)

        # update number of species seen so far
        self.metrics[species_id]["abundance_sum_species_counts"].append(
            self.metrics[species_id].abundance_current_total_species_count)
        self.metrics[species_id]["incidence_sum_species_counts"].append(
            self.metrics[species_id].incidence_current_total_species_count)

        # update degree of spatial aggregation
        self.metrics[species_id]["degree_of_co_occurrence"].append(self.metrics[species_id].current_co_occurrence)

        # update singleton and doubleton counts
        self.metrics[species_id]["abundance_singletons"].append(
            get_singletons(self.metrics[species_id].reference_sample_abundance))
        self.metrics[species_id]["incidence_singletons"].append(
            get_singletons(self.metrics[species_id].reference_sample_incidence))

        self.metrics[species_id]["abundance_doubletons"].append(
            get_doubletons(self.metrics[species_id].reference_sample_abundance))
        self.metrics[species_id]["incidence_doubletons"].append(
            get_doubletons(self.metrics[species_id].reference_sample_incidence))

        # update diversity profile
        if self.include_d0:
            self.__update_d0(species_id)
        if self.include_d1:
            self.__update_d1(species_id)
        if self.include_d2:
            self.__update_d2(species_id)

        # update completeness profile
        if self.include_c0:
            self.__update_c0(species_id)
        if self.include_c1:
            self.__update_c1(species_id)

        # update ACE profile
        if self.include_ace:
            self.__update_ace(species_id)

        # update ACE_modified profile
        if self.include_ace_modified:
            self.__update_ace_modified(species_id)

        # update ICE profile
        if self.include_ice:
            self.__update_ice(species_id)

        # update ICE_modified profile
        if self.include_ice_modified:
            self.__update_ice_modified(species_id)

        # update JACKKNIFE1_abundance profile
        if self.include_jackknife1_abundance:
            self.__update_jackknife1_abundance(species_id)

        # update JACKKNIFE1_incidence profile
        if self.include_jackknife1_incidence:
            self.__update_jackknife1_incidence(species_id)

        # update JACKKNIFE2_abundance profile
        if self.include_jackknife2_abundance:
            self.__update_jackknife2_abundance(species_id)

        # update JACKKNIFE2_incidence profile
        if self.include_jackknife2_incidence:
            self.__update_jackknife2_incidence(species_id)

        # update CHAO1 profile
        if self.include_chao1:
            self.__update_chao1(species_id)

        # update CHAO2 profile
        if self.include_chao2:
            self.__update_chao2(species_id)

        # update iCHAO1 profile
        if self.include_iChao1:
            self.__update_iChao1(species_id)

        # update iCHAO2 profile
        if self.include_iChao2:
            self.__update_iChao2(species_id)

        # update estimated sampling effort for target completeness
        for l in self.l_n:
            self.__update_l(l, species_id)

    def __update_d0(self, species_id: str) -> None:
        """
        updates D0 (=species richness) based on the current observations
        """
        # update sample metrics
        self.metrics[species_id]["abundance_sample_d0"].append(len(self.metrics[species_id].reference_sample_abundance))
        self.metrics[species_id]["incidence_sample_d0"].append(len(self.metrics[species_id].reference_sample_incidence))

        # update estimated metrics
        self.metrics[species_id]["abundance_estimate_d0"].append(
            hill_number_asymptotic(0, self.metrics[species_id].reference_sample_abundance,
                                   self.metrics[species_id].abundance_sample_size))
        self.metrics[species_id]["incidence_estimate_d0"].append(
            hill_number_asymptotic(0, self.metrics[species_id].reference_sample_incidence,
                                   self.metrics[species_id].incidence_sample_size, abundance=False))

        self.metrics[species_id]["incidence_estimate_d0_ci"].append(-1)

    ### Nina
    """def __update_ace(self, species_id: str) -> None:
        
        if self.include_ace:

            data = list(self.metrics[species_id].reference_sample_abundance.values())
            if not data:
                # Wenn keine Daten vorhanden sind, speichere None und beende
                self.metrics[species_id]["ace"].append(None)
                return

            S_abund = sum(1 for x in data if x > 10)  # häufige species
            S_rare_abund = sum(1 for x in data if x <= 10)  # rare species
            F1_abund = sum(1 for x in data if x == 1)  # singletons
            N_rare_abund = sum(x for x in data if x <= 10)  # Summe der Häufigkeiten rare species

            # calc von Fi_abund
            counts = Counter(data)
            Fi_abund = [counts[i] for i in range(1, 11)]  # Frequenzliste von 1 bis 10

            # calc ACE
            ace_result, gamma_sq_ace = ace(S_abund, S_rare_abund, F1_abund, N_rare_abund, Fi_abund)

            # if N_rare_abund = 0
            if N_rare_abund == 0:
                # wenn keine rare Spezies vorhanden --> return/ save S_obs
                self.metrics[species_id]["ace"].append(S_abund + S_rare_abund)
            else:
                # ACE result
                self.metrics[species_id]["ace"].append(ace_result)"""

    def __update_ace(self, species_id: str) -> None:
        if self.include_ace:
            data = [int(x) for x in self.metrics[species_id].reference_sample_abundance.values() if str(x).isdigit()]

            if not data:
                # if no data, append "none" and exit
                self.metrics[species_id]["ace"].append(None)
                return

            # parameters for the different species
            S_abund = sum(1 for x in data if x > 10)  # frequent species
            S_rare_abund = sum(1 for x in data if x <= 10)  # rare species
            F1_abund = sum(1 for x in data if x == 1)  # singletons
            N_rare_abund = sum(x for x in data if x <= 10)  # total count of rare species

            # frequencies of species with 1-10 observations
            counts = Counter(data)
            Fi_abund = [counts.get(i, 0) for i in range(1, 11)]

            # calculate ACE
            ace_result, gamma_sq_ace = ace(S_abund, S_rare_abund, F1_abund, N_rare_abund, Fi_abund)

            if N_rare_abund == 0:
                # if no rare species, return observed species count
                self.metrics[species_id]["ace"].append(S_abund + S_rare_abund)
            else:
                self.metrics[species_id]["ace"].append(ace_result)

    def __update_ace_modified(self, species_id: str) -> None:
        if self.include_ace_modified:
            data = [int(x) for x in self.metrics[species_id].reference_sample_abundance.values() if str(x).isdigit()]

            if not data:
                # append None if no data
                self.metrics[species_id].setdefault("ace_modified", []).append(None)
                return

            # parameters for the different species
            S_abund = sum(1 for x in data if x > 10)  # frequent species
            S_rare_abund = sum(1 for x in data if x <= 10)  # rare species
            F1_abund = sum(1 for x in data if x == 1)  # singletons
            N_rare_abund = sum(x for x in data if x <= 10)  # total count of rare species

            # frequencies of species with 1-10 observations
            counts = Counter(data)
            Fi_abund = [counts.get(i, 0) for i in range(1, 11)]

            if N_rare_abund > 0:
                # calculate ACE_modified
                ace_result_modified, gamma_sq_ace_modified = ace_modified(S_abund, S_rare_abund, F1_abund, N_rare_abund,
                                                                          Fi_abund)
                self.metrics[species_id]["ace_modified"].append(ace_result_modified)
            else:
                # if no rare species, append observed species count
                S_obs = len([x for x in data if x > 0])
                self.metrics[species_id]["ace_modified"].append(S_obs)

    """def __update_ace_modified(self, species_id: str) -> None:
        if self.include_ace_modified:
            # Daten aus der Referenzstichprobe
            data = list(self.metrics[species_id].reference_sample_abundance.values())

            # Abbruch, wenn keine Daten vorhanden sind
            if not data:
                self.metrics[species_id].setdefault("ace_modified", []).append(None)
                return

            # Berechnung der benötigten Werte
            S_abund = sum(1 for x in data if x > 10)  # Häufige Arten
            S_rare_abund = sum(1 for x in data if x <= 10)  # Seltene Arten
            F1_abund = sum(1 for x in data if x == 1)  # Singletons
            N_rare_abund = sum(x for x in data if x <= 10)  # Summe der Häufigkeiten seltener Arten

            # Frequenzliste F_i für i = 1 bis 10
            counts = Counter(data)
            Fi_abund = [counts[i] for i in range(1, 11)]

            # Berechnung der modifizierten ACE-Schätzung
            ace_result_modified, gamma_sq_ace_modified = ace_modified(S_abund, S_rare_abund, F1_abund, N_rare_abund,
                                                                      Fi_abund)

            # Ergebnis speichern
            if N_rare_abund == 0:
                # Wenn keine seltenen Arten vorhanden sind, speichere nur die Summe der beobachteten Arten
                self.metrics[species_id].setdefault("ace_modified", []).append(S_abund + S_rare_abund)
            else:
                # Speichere das modifizierte ACE-Ergebnis
                self.metrics[species_id].setdefault("ace_modified", []).append(ace_result_modified)"""

    """"# angepasst
    def __update_ace_modified(self, species_id: str) -> None:
        
        if self.include_ace_modified:
            data = list(self.metrics[species_id].reference_sample_abundance.values())

            S_abund = sum(1 for x in data if x > 10)  # häufige species
            S_rare_abund = sum(1 for x in data if x <= 10)  # rare species
            F1_abund = sum(1 for x in data if x == 1)  # singletons
            N_rare_abund = sum(x for x in data if x <= 10)  # Gesamtzahl der rare species

            # Häufigkeiten von species bis 10 Beobachtungen (Fi)
            counts = Counter(data)
            Fi_abund = [counts[i] for i in range(1, 11)]

            if N_rare_abund > 0:  # es existieren rare species
                # calc von C_ace
                C_ace = calculate_C_ace(F1_abund, N_rare_abund)

                if C_ace > 0:  # Regulärer ACE-Schätzer möglich
                    ace_modified_result, gamma_sq_ace_modified = ace_modified(
                        S_abund, S_rare_abund, F1_abund, N_rare_abund, Fi_abund
                    )
                    self.metrics[species_id]["ace_modified"].append(ace_modified_result)
                else:  # Coverage = 0, return S_obs
                    S_obs = len([x for x in data if x > 0])
                    self.metrics[species_id]["ace_modified"].append(S_obs)
            else:  # keine rare species, return S_obs (die beobachteten)
                S_obs = len([x for x in data if x > 0])
                self.metrics[species_id]["ace_modified"].append(S_obs)"""

    """def __update_ice(self, species_id: str) -> None:
       
        if self.include_ice:
            # 1. Daten aus der Referenzstichprobe der Abundanz-Daten holen
            species_counts = list(self.metrics[species_id].reference_sample_incidence.values())

            if not species_counts:
                self.metrics[species_id].setdefault("ice", []).append(None)
                return

            # 2. Berechnungen der benötigten Werte
            S_freq = sum(1 for x in species_counts if x > 10)  # Häufige Arten
            S_inf = sum(1 for x in species_counts if x <= 10)  # Seltene Arten
            N_inf = sum(x for x in species_counts if x <= 10)  # Gesamtzahl der seltenen Arten
            Q1 = sum(1 for x in species_counts if x == 1)  # Singletons (Arten, die nur einmal vorkommen)
            Qj = [sum(1 for x in species_counts if x == j) for j in range(1, 11)]

            # Berechnung von m_inf: Anzahl der Stichproben mit seltenen Arten
            m_inf = sum(1 for sample in self.metrics[species_id].reference_sample_incidence.values() if sample <= 10)

            # Berechnung von Qj: Häufigkeit der Arten, die j-mal vorkommen
            counts = Counter(species_counts)
            Qj = [sum(1 for x in species_counts if x == j) for j in range(1, 11)]

            # 3. Berechnung von C_ice und gamma^2_ice
            C_ice = calculate_C_ice(Q1, N_inf)
            gamma_sq_ice = calculate_gamma_sq_ice(S_inf, C_ice, m_inf, N_inf, Qj)

            # 4. Berechnung des ICE-Ergebnisses
            if C_ice == 0:
                # Wenn C_ice = 0, dann nur die Summe der häufigen und seltenen Arten speichern
                self.metrics[species_id].setdefault("ice", []).append(S_freq + S_inf)
            else:
                # Berechnung des ICE-Ergebnisses mit modifiziertem gamma^2_ice
                ice_result, gamma_sq_ice_result = ice(S_freq, S_inf, Q1, N_inf, Qj, m_inf)
                self.metrics[species_id].setdefault("ice", []).append(ice_result)"""

    def __update_ice(self, species_id: str) -> None:
        if self.include_ice:
            species_counts = [int(x) for x in self.metrics[species_id].reference_sample_incidence.values() if
                              str(x).isdigit()]

            if not species_counts:
                self.metrics[species_id].setdefault("ice", []).append(None)
                return

            # parameters for the different species
            S_freq = sum(1 for x in species_counts if x > 10)  # frequent species
            S_inf = sum(1 for x in species_counts if x <= 10)  # rare species
            Q1 = sum(1 for x in species_counts if x == 1)  # singletons
            N_inf = sum(x for x in species_counts if x <= 10)  # total count of rare species

            # number of samples with rare species
            m_inf = sum(1 for sample in species_counts if sample <= 10)

            # frequencies of species appearing 1 to 10 times
            Qj = [sum(1 for x in species_counts if x == j) for j in range(1, 11)]

            C_ice = calculate_C_ice(Q1, N_inf)

            if N_inf > 0 and C_ice > 0:
                ice_result, _ = ice(S_freq, S_inf, Q1, N_inf, Qj, m_inf)
                self.metrics[species_id]["ice"].append(ice_result)
            else:
                # if no rare species or invalid coverage, append observed species count
                S_obs = S_freq + S_inf
                self.metrics[species_id]["ice"].append(S_obs)

    def __update_ice_modified(self, species_id: str) -> None:
        if self.include_ice_modified:
            species_counts = [int(x) for x in self.metrics[species_id].reference_sample_incidence.values() if
                              str(x).isdigit()]

            if not species_counts:
                self.metrics[species_id].setdefault("ice_modified", []).append(None)
                return

            # parameters for the different species
            S_freq = sum(1 for x in species_counts if x > 10)  # frequent species
            S_inf = sum(1 for x in species_counts if x <= 10)  # rare species
            Q1 = sum(1 for x in species_counts if x == 1)  # singletons
            N_inf = sum(x for x in species_counts if x <= 10)  # total count of rare species

            # number of samples with rare species
            m_inf = sum(1 for sample in species_counts if sample <= 10)

            # frequencies of species appearing 1 to 10 times
            Qj = [sum(1 for x in species_counts if x == j) for j in range(1, 11)]

            C_ice = calculate_C_ice(Q1, N_inf)

            if N_inf > 0 and C_ice > 0:
                ice_modified_result, _ = ice_modified(S_freq, S_inf, Q1, N_inf, Qj, m_inf)
                self.metrics[species_id]["ice_modified"].append(ice_modified_result)
            else:
                # if no rare species or invalid coverage, append observed species count
                S_obs = S_freq + S_inf
                self.metrics[species_id]["ice_modified"].append(S_obs)

    """def __update_ice(self, species_id: str) -> None:
        
        if self.include_ice:
            species_counts = list(self.metrics[species_id].reference_sample_incidence.values())

            if not species_counts:
                self.metrics[species_id].setdefault("ice", []).append(None)
                return

            S_freq = sum(1 for x in species_counts if x > 10)
            S_inf = sum(1 for x in species_counts if x <= 10)  # seltene species
            Q1 = sum(1 for x in species_counts if x == 1)
            N_inf = sum(x for x in species_counts if x <= 10)  # gesamtzahl der seltenen Arten

            # m_inf: # seltene species
            m_inf = sum(1 for sample in self.metrics[species_id].reference_sample_incidence.values() if sample <= 10)

            Qj = [sum(1 for x in species_counts if x == j) for j in range(1, 11)]

            C_ice = calculate_C_ice(Q1, N_inf)
            gamma_sq_ice = calculate_gamma_sq_ice(S_inf, C_ice, m_inf, N_inf, Qj)

            if N_inf > 0:
                # if S_inf --> calc ice
                ice_result, _ = ice(S_freq, S_inf, Q1, N_inf, Qj, m_inf)
                self.metrics[species_id]["ice"].append(ice_result)
            else:
                # if no S_inf species --> save S_obs
                S_obs = S_freq + S_inf
                self.metrics[species_id]["ice"].append(S_obs)"""

    """# update methode angepasst --> statt none, return observed species
    def __update_ice_modified(self, species_id: str) -> None:
        
        if self.include_ice_modified:
            data = list(self.metrics[species_id].reference_sample_incidence.values())

            S_freq = sum(1 for x in data if x > 10)
            S_inf = sum(1 for x in data if x <= 10)
            Q1 = sum(1 for x in data if x == 1)
            N_inf = sum(x for x in data if x <= 10)
            Qj = [sum(1 for x in data if x == i) for i in range(1, 11)]
            m_inf = sum(1 for sample in self.metrics[species_id].reference_sample_incidence.values() if sample <= 10)

            if N_inf > 0:
                ice_modified_result, gamma_sq_ice_modified = ice_modified(S_freq, S_inf, Q1, N_inf, Qj, m_inf)
                self.metrics[species_id]["ice_modified"].append(ice_modified_result)
            else:
                S_obs = len([x for x in data if x > 0])
                self.metrics[species_id]["ice_modified"].append(S_obs)"""


    """def __update_jackknife1_abundance(self, species_id: str) -> None:
        if self.include_jackknife1_abundance:
            S_obs = len(self.metrics[species_id].reference_sample_abundance)
            Q1 = sum(1 for count in self.metrics[species_id].reference_sample_abundance.values() if count == 1)
            m = self.metrics[species_id].abundance_sample_size

            jackknife1_abundance_result = jackknife1_abundance(S_obs, Q1, m)
            #self.metrics[species_id]["jackknife1_abundance"].append(jackknife1_abundance_result)
            self.metrics[species_id].setdefault("jackknife1_abundance", []).append(jackknife1_abundance_result)"""

    def __update_jackknife1_abundance(self, species_id: str) -> None:
        if self.include_jackknife1_abundance:
            S_obs = len(self.metrics[species_id].reference_sample_abundance)
            f1 = sum(1 for count in self.metrics[species_id].reference_sample_abundance.values() if count == 1)

            jackknife1_abundance_result = jackknife1_abundance(S_obs, f1)
            self.metrics[species_id].setdefault("jackknife1_abundance", []).append(jackknife1_abundance_result)

    def __update_jackknife1_incidence(self, species_id: str) -> None:
        if self.include_jackknife1_incidence:
            S_obs = len(self.metrics[species_id].reference_sample_incidence)
            Q1 = sum(1 for count in self.metrics[species_id].reference_sample_incidence.values() if count == 1)
            m = self.metrics[species_id].incidence_sample_size

            jackknife1_incidence_result = jackknife1_incidence(S_obs, Q1, m)
            self.metrics[species_id].setdefault("jackknife1_incidence", []).append(jackknife1_incidence_result)

    def __update_jackknife2_abundance(self, species_id: str) -> None:
        if self.include_jackknife2_abundance:
            S_obs = len(self.metrics[species_id].reference_sample_abundance)
            f1 = sum(1 for count in self.metrics[species_id].reference_sample_abundance.values() if count == 1)
            f2 = sum(1 for count in self.metrics[species_id].reference_sample_abundance.values() if count == 2)

            jackknife2_abundance_result = jackknife2_abundance(S_obs, f1, f2)
            self.metrics[species_id].setdefault("jackknife2_abundance", []).append(jackknife2_abundance_result)

    def __update_jackknife2_incidence(self, species_id: str) -> None:
        if self.include_jackknife2_incidence:
            S_obs = len(self.metrics[species_id].reference_sample_incidence)
            Q1 = sum(1 for count in self.metrics[species_id].reference_sample_incidence.values() if count == 1)
            Q2 = sum(1 for count in self.metrics[species_id].reference_sample_incidence.values() if count == 2)
            m = self.metrics[species_id].incidence_sample_size

            jackknife2_incidence_result = jackknife2_incidence(S_obs, Q1, Q2, m)
            self.metrics[species_id].setdefault("jackknife2_incidence", []).append(jackknife2_incidence_result)


    """def __update_chao1(self, species_id: str) -> None:
        if self.include_chao1:
            data = list(self.metrics[species_id].reference_sample_abundance.values())
            S_obs = sum(1 for x in data if x > 0)
            f1 = sum(1 for x in data if x == 1)
            f2 = sum(1 for x in data if x == 2)

            if f2 == 0:
                chao1_estimate = S_obs + (f1 * (f1 - 1)) / 2
            else:
                chao1_estimate = S_obs + (f1 ** 2) / (2 * f2)
            #self.metrics[species_id]["chao1"].append(chao1_estimate)
            self.metrics[species_id].setdefault("chao1", []).append(chao1_estimate)"""

    def __update_chao1(self, species_id: str) -> None:
        data = list(self.metrics[species_id].reference_sample_abundance.values())
        if not data:
            self.metrics[species_id]["chao1"].append(None)
            return

        S_obs = sum(1 for x in data if x > 0)
        f1 = sum(1 for x in data if x == 1)
        f2 = sum(1 for x in data if x == 2)

        if f2 == 0:
            chao1_estimate = S_obs + (f1 * (f1 - 1)) / 2
        else:
            chao1_estimate = S_obs + (f1 ** 2) / (2 * f2)

        self.metrics[species_id]["chao1"].append(chao1_estimate)

    """def __update_chao2(self, species_id: str) -> None:
        if self.include_chao2:
            #data = list(self.metrics[species_id].reference_sample_abundance.values())
            data = list(self.metrics[species_id].reference_sample_incidence.values())
            S_obs = sum(1 for x in data if x > 0)
            Q1 = sum(1 for x in data if x == 1)
            Q2 = sum(1 for x in data if x == 2)

            if Q2 == 0:
                chao2_estimate = S_obs + (Q1 * (Q1 - 1)) / 2
            else:
                #chao2_estimate = S_obs + (Q1 * Q1) / (2 * Q2)
                chao2_estimate = S_obs + (Q1 ** 2) / (2 * Q2)
            # save
            #if "chao2" not in self.metrics[species_id]:
                #self.metrics[species_id]["chao2"] = []
            #self.metrics[species_id]["chao2"].append(chao2_estimate)
            self.metrics[species_id].setdefault("chao2", []).append(chao2_estimate)"""

    def __update_chao2(self, species_id: str) -> None:
        data = list(self.metrics[species_id].reference_sample_incidence.values())
        if not data:
            self.metrics[species_id]["chao2"].append(None)
            return

        S_obs = sum(1 for x in data if x > 0)
        Q1 = sum(1 for x in data if x == 1)
        Q2 = sum(1 for x in data if x == 2)

        if Q2 == 0:
            chao2_estimate = S_obs + (Q1 * (Q1 - 1)) / 2
        else:
            chao2_estimate = S_obs + (Q1 ** 2) / (2 * Q2)

        self.metrics[species_id]["chao2"].append(chao2_estimate)

    """def __update_iChao1(self, species_id: str) -> None:
        if self.include_iChao1:
            data = list(self.metrics[species_id].reference_sample_abundance.values())
            S_obs = sum(1 for x in data if x > 0)
            f1 = sum(1 for x in data if x == 1)
            f2 = sum(1 for x in data if x == 2)
            f3 = sum(1 for x in data if x == 3)
            f4 = sum(1 for x in data if x == 4)

            if S_obs > 0:
                # calculate chao1 first
                S_chao1 = chao1(S_obs, f1, f2)
                #iChao1_result = iChao1(S_obs, f1, f2, f3, f4)
                iChao1_result = iChao1(S_chao1, f1, f2, f3, f4)
                #self.metrics[species_id]["iChao1"].append(iChao1_result)
                self.metrics[species_id].setdefault("iChao1", []).append(iChao1_result)"""

    def __update_iChao1(self, species_id: str) -> None:
        data = list(self.metrics[species_id].reference_sample_abundance.values())
        if not data:
            self.metrics[species_id]["iChao1"].append(None)
            return

        # frequency counts (from f1 to f4)
        S_obs = sum(1 for x in data if x > 0) # number of observed species
        f1 = sum(1 for x in data if x == 1)
        f2 = sum(1 for x in data if x == 2)
        f3 = sum(1 for x in data if x == 3)
        f4 = sum(1 for x in data if x == 4)

        # calculation
        S_chao1 = chao1(S_obs, f1, f2)
        iChao1_result = iChao1(S_chao1, f1, f2, f3, f4)

        self.metrics[species_id]["iChao1"].append(iChao1_result)

    """def __update_iChao2(self, species_id: str) -> None:
        if self.include_iChao2:
            data = list(self.metrics[species_id].reference_sample_incidence.values())
            S_obs = sum(1 for x in data if x > 0)
            Q1 = sum(1 for x in data if x == 1)
            Q2 = sum(1 for x in data if x == 2)
            Q3 = sum(1 for x in data if x == 3)
            Q4 = sum(1 for x in data if x == 4)
            T = sum(data)

            if T > 0:
                # calculate chao2 first
                S_chao2 = chao2(S_obs, Q1, Q2)
                #iChao2_result = iChao2(S_obs, Q1, Q2, Q3, Q4, T)
                iChao2_result = iChao2(S_chao2, Q1, Q2, Q3, Q4, T)
                #self.metrics[species_id]["iChao2"].append(iChao2_result)
                self.metrics[species_id].setdefault("iChao2", []).append(iChao2_result)
                # die return Werte sind hier optional
                return iChao2_result
            return 0.0  # if T = 0"""

    def __update_iChao2(self, species_id: str) -> None:
        data = list(self.metrics[species_id].reference_sample_incidence.values())
        if not data:
            self.metrics[species_id]["iChao2"].append(None)
            return

        # frequency counts (from Q1 to Q4)
        S_obs = sum(1 for x in data if x > 0)
        Q1 = sum(1 for x in data if x == 1)
        Q2 = sum(1 for x in data if x == 2)
        Q3 = sum(1 for x in data if x == 3)
        Q4 = sum(1 for x in data if x == 4)
        T = sum(data) # total number of sampling units

        # calculation
        if T > 0:
            S_chao2 = chao2(S_obs, Q1, Q2)
            iChao2_result = iChao2(S_chao2, Q1, Q2, Q3, Q4, T)
            self.metrics[species_id]["iChao2"].append(iChao2_result)
        else:
            self.metrics[species_id]["iChao2"].append(S_obs)

        ### Nina

    def __update_d1(self, species_id: str) -> None:
        """
        updates D1 (=exponential of Shannon entropy) based on the current observations
        """
        # update sample metrics
        self.metrics[species_id]["abundance_sample_d1"].append(
            entropy_exp(self.metrics[species_id].reference_sample_abundance))
        self.metrics[species_id]["incidence_sample_d1"].append(
            entropy_exp(self.metrics[species_id].reference_sample_incidence))

        # update estimated metrics
        self.metrics[species_id]["abundance_estimate_d1"].append(
            hill_number_asymptotic(1, self.metrics[species_id].reference_sample_abundance,
                                   self.metrics[species_id].abundance_sample_size))
        self.metrics[species_id]["incidence_estimate_d1"].append(
            hill_number_asymptotic(1, self.metrics[species_id].reference_sample_incidence,
                                   self.metrics[species_id].incidence_sample_size, abundance=False))

        self.metrics[species_id]["incidence_estimate_d1_ci"].append(-1)

    def __update_d2(self, species_id: str) -> None:
        """
        updates D2 (=Simpson Diversity Index) based on the current observations
        """
        # update sample metrics
        self.metrics[species_id]["abundance_sample_d2"].append(
            simpson_diversity(self.metrics[species_id].reference_sample_abundance))
        self.metrics[species_id]["incidence_sample_d2"].append(
            simpson_diversity(self.metrics[species_id].reference_sample_incidence))

        #  update estimated metrics
        self.metrics[species_id]["abundance_estimate_d2"].append(
            hill_number_asymptotic(2, self.metrics[species_id].reference_sample_abundance,
                                   self.metrics[species_id].abundance_sample_size))
        self.metrics[species_id]["incidence_estimate_d2"].append(
            hill_number_asymptotic(2, self.metrics[species_id].reference_sample_incidence,
                                   self.metrics[species_id].incidence_sample_size, abundance=False))

        self.metrics[species_id]["incidence_estimate_d2_ci"].append(-1)

    def __update_c0(self, species_id: str) -> None:
        """
        updates C0 (=completeness) based on the current observations
        """
        self.metrics[species_id]["abundance_c0"].append(
            completeness(self.metrics[species_id].reference_sample_abundance))
        self.metrics[species_id]["incidence_c0"].append(
            completeness(self.metrics[species_id].reference_sample_incidence))

        self.metrics[species_id]["incidence_c0_ci"].append(-1)

    def __update_c1(self, species_id: str) -> None:
        """
        updates C1 (=coverage) based on the current observations
        """
        self.metrics[species_id]["abundance_c1"].append(
            coverage(self.metrics[species_id].reference_sample_abundance,
                     self.metrics[species_id].abundance_sample_size))
        self.metrics[species_id]["incidence_c1"].append(
            coverage(self.metrics[species_id].reference_sample_incidence,
                     self.metrics[species_id].incidence_sample_size))

        self.metrics[species_id]["incidence_c1_ci"].append(-1)

    def __update_l(self, g: float, species_id: str) -> None:
        """
        updates l_g (=expected number additional observations for reaching completeness g) based on the current
        observations
        :param g: desired  completeness
        """
        self.metrics[species_id]["abundance_l_" + str(g)].append(
            sampling_effort_abundance(g, self.metrics[species_id].reference_sample_abundance,
                                      self.metrics[species_id].abundance_sample_size))
        self.metrics[species_id]["incidence_l_" + str(g)].append(
            sampling_effort_incidence(g, self.metrics[species_id].reference_sample_incidence,
                                      self.metrics[species_id].incidence_sample_size))

    def summarize(self, species_id: str = None) -> None:
        """
        prints a summary of the species profiles of the current reference sample.

        """
        species_ids = self.metrics.keys() if species_id is None else [species_id]

        for species_id in species_ids:
            print("### "+species_id+" ###")
            print("%-25s %-20s %s" % ("Sample Stats", "Abundance", "Incidence"))
            print("%-25s %-20s %s" % ("", "---------", "---------"))
            print("%-25s %-20s %s" % ("No Observations", str(self.metrics[species_id]["abundance_no_observations"][-1]),str(self.metrics[species_id]["incidence_no_observations"][-1])))
            print("%-25s %-20s %s" % ("No Species", str(self.metrics[species_id]["abundance_sum_species_counts"][-1]),str(self.metrics[species_id]["incidence_sum_species_counts"][-1])))

            print("%-25s %-20s %s" % ("Singletons", str(self.metrics[species_id]["abundance_singletons"][-1]),str(self.metrics[species_id]["abundance_doubletons"][-1])))
            print("%-25s %-20s %s" % ("Doubletons", str(self.metrics[species_id]["abundance_doubletons"][-1]),str(self.metrics[species_id]["incidence_doubletons"][-1])))
            print("%-25s %s" % ("Degree of Co-Occurrence", str(self.metrics[species_id]["degree_of_co_occurrence"][-1])))
            print()
            print("%-25s %-20s %-20s %s" % ("Abundance:", "Observed", "Estimate", "Stdev"))
            print("%-25s %-20s %-20s %s" % ("", "--------", "--------", "-----"))
            print("%-25s %-20s %-20s %s" % ("D0", str(self.metrics[species_id]["abundance_sample_d0"][-1]), str(self.metrics[species_id]["abundance_estimate_d0"][-1]), "-"))
            print("%-25s %-20s %-20s %s" % ("D1", str(self.metrics[species_id]["abundance_sample_d1"][-1]), str(self.metrics[species_id]["abundance_estimate_d1"][-1]), "-"))
            print("%-25s %-20s %-20s %s" % ("D2", str(self.metrics[species_id]["abundance_sample_d2"][-1]), str(self.metrics[species_id]["abundance_estimate_d2"][-1]), "-"))
            print("%-25s %-20s %-20s %s" % ("C0", "-", str(self.metrics[species_id]["abundance_c0"][-1]), "-"))
            print("%-25s %-20s %-20s %s" % ("C1", "-", str(self.metrics[species_id]["abundance_c1"][-1]), "-"))

            for l in self.l_n:
                print("%-25s %-20s %-20s %s" %("l_"+str(l), "-", str(self.metrics[species_id]["abundance_l_" + str(l)][-1]), "-"))
            print()
            print("%-25s %-20s %-20s %s" % ("Incidence:", "Observed", "Estimate", "Stdev"))
            print("%-25s %-20s %-20s %s" % ("", "--------", "--------", "-----"))
            print("%-25s %-20s %-20s %s" % ("D0", str(self.metrics[species_id]["incidence_sample_d0"][-1]), str(self.metrics[species_id]["incidence_estimate_d0"][-1]),str(self.metrics[species_id]["incidence_estimate_d0_ci"][-1]) if self.no_bootstrap_samples>0 else "-"))
            print("%-25s %-20s %-20s %s" % ("D1", str(self.metrics[species_id]["incidence_sample_d1"][-1]), str(self.metrics[species_id]["incidence_estimate_d1"][-1]),str(self.metrics[species_id]["incidence_estimate_d1_ci"][-1]) if self.no_bootstrap_samples>0 else "-"))
            print("%-25s %-20s %-20s %s" % ("D2", str(self.metrics[species_id]["incidence_sample_d2"][-1]), str(self.metrics[species_id]["incidence_estimate_d2"][-1]),str(self.metrics[species_id]["incidence_estimate_d2_ci"][-1]) if self.no_bootstrap_samples>0 else "-"))
            print("%-25s %-20s %-20s %s" % ("C0", "-", str(self.metrics[species_id]["incidence_c0"][-1]),str(self.metrics[species_id]["incidence_c0_ci"][-1]) if self.no_bootstrap_samples > 0 else "-"))
            print("%-25s %-20s %-20s %s" % ("C1", "-", str(self.metrics[species_id]["incidence_c1"][-1]),str(self.metrics[species_id]["incidence_c1_ci"][-1]) if self.no_bootstrap_samples > 0 else "-"))
            for l in self.l_n:
                print("%-25s %-20s %-20s %s" % ("l_"+str(l), "-", str(self.metrics[species_id]["incidence_l_" + str(l)][-1]), "-"))
            print("")
            print("")


    @deprecated
    def print_metrics(self) -> None:
        """
        prints the Diversity and Completeness Profile of the current observations
        """
        for species_id in self.species_retrieval:
            print("### " + species_id + " ###")
            print("### SAMPLE STATS ###")
            print("Abundance")
            print("%-30s %s" % ("     No Observations:", str(self.metrics[species_id]["abundance_no_observations"])))
            print("%-30s %s" % (
                "     Total Species Count:", str(self.metrics[species_id]["abundance_sum_species_counts"])))
            print("%-30s %s" % ("     Singletons:", str(self.metrics[species_id]["abundance_singletons"])))
            print("%-30s %s" % ("     Doubletons:", str(self.metrics[species_id]["abundance_doubletons"])))

            print("Incidence")
            print("%-30s %s" % ("     No Observations:", str(self.metrics[species_id]["incidence_no_observations"])))
            print("%-30s %s" % (
                "     Total Species Count:", str(self.metrics[species_id]["incidence_sum_species_counts"])))
            print("%-30s %s" % ("     Singletons:", str(self.metrics[species_id]["incidence_singletons"])))
            print("%-30s %s" % ("     Doubletons:", str(self.metrics[species_id]["incidence_doubletons"])))

            print("%-30s %s" % ("     Empty Traces:", str(self.metrics[species_id].empty_traces)))
            print("%-30s %s" % ("Degree of Co-Occurrence:", str(self.metrics[species_id]["degree_of_aggregation"])))
            print()
            print("### DIVERSITY AND COMPLETENESS PROFILE ###")
            print("Abundance")
            if self.include_d0:
                print("%-30s %s" % ("     D0 - sample:", (self.metrics[species_id]["abundance_sample_d0"])))
                print("%-30s %s" % ("     D0 - estimate:", str(self.metrics[species_id]["abundance_estimate_d0"])))

            if self.include_ace:
                print("%-30s %s" % ("ace:", str(self.metrics[species_id]["ace_abundance_estimate_d0"])))

            if self.include_ace_modified:
                print("%-30s %s" % ("ace_modified:", str(self.metrics[species_id]["ace_modified_abundance_estimate_d0"])))

            if self.include_chao1:
                print("%-30s %s" % ("chao1:", str(self.metrics[species_id]["chao1_abundance_estimate_d0"])))

            if self.include_iChao1:
                print("%-30s %s" % ("iChao1:", str(self.metrics[species_id]["iChao1_abundance_estimate_d0"])))

            if self.include_jackknife1_abundance:
                print("%-30s %s" % ("jackknife1_abundance:", str(self.metrics[species_id]["jackknife1_abundance_estimate_d0"])))

            if self.include_jackknife2_abundance:
                print("%-30s %s" % ("jackknife2_abundance:", str(self.metrics[species_id]["jackknife2_abundance_estimate_d0"])))

            if self.include_d1:
                print("%-30s %s" % ("     D1 - sample:", str(self.metrics[species_id]["abundance_sample_d1"])))
                print("%-30s %s" % ("     D1 - estimate:", str(self.metrics[species_id]["abundance_estimate_d1"])))
            if self.include_d2:
                print("%-30s %s" % ("     D2 - sample:", str(self.metrics[species_id]["abundance_sample_d2"])))
                print("%-30s %s" % ("     D2 - estimate:", str(self.metrics[species_id]["abundance_estimate_d2"])))
            if self.include_c0:
                print("%-30s %s" % ("     C0:", str(self.metrics[species_id]["abundance_c0"])))
            if self.include_c1:
                print("%-30s %s" % ("     C1:", str(self.metrics[species_id]["abundance_c1"])))
            for l in self.l_n:
                print("%-30s %s" % ("     l_" + str(l) + ":", str(self.metrics[species_id]["abundance_l_" + str(l)])))
            print("Incidence")
            if self.include_d0:
                print("%-30s %s" % ("     D0 - sample:", (self.metrics[species_id]["incidence_sample_d0"])))
                print("%-30s %s" % ("     D0 - estimate:", str(self.metrics[species_id]["incidence_estimate_d0"])))
                print("%-30s %s" % ("     D0 - CI:", str(self.metrics[species_id]["incidence_estimate_d0_ci"])))

            if self.include_ice:
                print("%-30s %s" % ("ice:", str(self.metrics[species_id]["ice_incidence_estimate_d0"])))

            if self.include_ice_modified:
                print("%-30s %s" % ("ice_modified:", str(self.metrics[species_id]["ice_modified_incidence_estimate_d0"])))

            if self.include_jackknife1_incidence:
                print("%-30s %s" % ("jackknife1_incidence:", str(self.metrics[species_id]["jackknife1_incidence_estimate_d0"])))

            if self.include_jackknife2_incidence:
                print("%-30s %s" % ("jackknife2_incidence:", str(self.metrics[species_id]["jackknife2_incidence_estimate_d0"])))

            if self.include_chao2:
                print("%-30s %s" % ("chao2:", str(self.metrics[species_id]["chao2_incidence_estimate_d0"])))

            if self.include_iChao2:
                print("%-30s %s" % ("iChao2:", str(self.metrics[species_id]["iChao2_incidence_estimate_d0"])))

            if self.include_d1:
                print("%-30s %s" % ("     D1 - sample:", str(self.metrics[species_id]["incidence_sample_d1"])))
                print("%-30s %s" % ("     D1 - estimate:", str(self.metrics[species_id]["incidence_estimate_d1"])))
                print("%-30s %s" % ("     D1 - CI:", str(self.metrics[species_id]["incidence_estimate_d1_ci"])))

            if self.include_d2:
                print("%-30s %s" % ("     D2 - sample:", str(self.metrics[species_id]["incidence_sample_d2"])))
                print("%-30s %s" % ("     D2 - estimate:", str(self.metrics[species_id]["incidence_estimate_d2"])))
                print("%-30s %s" % ("     D2 - CI:", str(self.metrics[species_id]["incidence_estimate_d2_ci"])))

            if self.include_c0:
                print("%-30s %s" % ("     C0:", str(self.metrics[species_id]["incidence_c0"])))
                print("%-30s %s" % ("     C0 - CI:", str(self.metrics[species_id]["incidence_c0_ci"])))

            if self.include_c1:
                print("%-30s %s" % ("     C1:", str(self.metrics[species_id]["incidence_c1"])))
                print("%-30s %s" % ("     C1 - CI:", str(self.metrics[species_id]["incidence_c1_ci"])))

            for l in self.l_n:
                print("%-30s %s" % ("     l_" + str(l) + ":", str(self.metrics[species_id]["incidence_l_" + str(l)])))
            print()

    def to_dataFrame(self, include_all=True) -> DataFrame:
        """
        returns the diversity and completeness profile of the current observations as a data frame
        :returns: a data frame view of the Diversity and Completeness Profile
        """
        return pd.DataFrame([[i, j, ix, v]
                             for i in self.metrics.keys()
                             for j in self.metrics[i].keys()
                             for ix, v in enumerate(self.metrics[i][j])
                             ], columns=["species", "metric", "observation", "value"]
                            ) if include_all else pd.DataFrame([[i, j, self.metrics[i][j][-1]]
                                                                for i in self.metrics.keys()
                                                                for j in self.metrics[i].keys()
                                                                ], columns=["species", "metric", "value"])
    
    
