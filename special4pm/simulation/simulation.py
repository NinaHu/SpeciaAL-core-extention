import pm4py
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from tqdm import tqdm


def simulate_model(petri_net, im, fm, size, log=None):
    """
    creates a log of size <no_traces> adhering to the behaviour described by the provided net. If a log is provided
    transition probabilities are chosen according to the stochastic map built from the provided log.
    :param size: size of the returned event log
    :param petri_net: a petri net
    :param im: initial marking of the petri net
    :param fm: final marking of the petri net
    :param log: event log determining transition probabilities
    :return: event log
    """
    log = pm4py.algo.simulation.playout.petri_net.algorithm.apply(petri_net, im, fm, {'noTraces': size,  'maxTraceLength':1000},
                                                                  variant=pm4py.algo.simulation.playout.petri_net.algorithm.Variants.BASIC_PLAYOUT)
    for i in range(len(log)):
        while len(log[i]) == 0:
            tr = pm4py.algo.simulation.playout.petri_net.algorithm.apply(petri_net, im, fm, {'noTraces': 1, 'maxTraceLength':1000},
                                                                                      variant=pm4py.algo.simulation.playout.petri_net.algorithm.Variants.BASIC_PLAYOUT)[0]
            log[i] = tr
        if len(log[i]) == 0:
            print("ZEROLENGTH TRACE")
    return log
