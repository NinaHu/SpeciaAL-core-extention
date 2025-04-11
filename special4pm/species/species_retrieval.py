import math
from datetime import datetime, timedelta

import pm4py


def retrieve_species_n_gram(trace, n):
    """
    TODO: this return a list of exactly n-grams, i.e. if a sequence is less than n-grams, an empty list is returned
    :param trace:
    :param n:
    :return:
    """
    #if len(trace) < n:
    #    return ["NULL"]
    events = [x['concept:name'] for x in trace]
    #if len(events) <n:
    #    [events.append("NULL") for _ in range(n-len(events))]
    #if n >= 2:
    #    events.insert(0, "START")
    #    events.append("END")
    return [",".join(events[x:x + n]) for x in range(0, len(events) - n+1)]


def retrieve_species_trace_variant(trace):
    return [",".join([x["concept:name"] for x in trace])]


def retrieve_timed_activity(trace, interval_size):
    l=[]
    if "lifecycle:transition" in trace[0]:
        open_set = []
        for ide,e in enumerate(trace):
            if "start" in e["lifecycle:transition"].lower():
                open_set.append(e)
            elif "complete" in e["lifecycle:transition"].lower():
                found = False
                for idx, x in enumerate(open_set):
                    if x["concept:name"] == e["concept:name"]:
                        time = trace[ide]["time:timestamp"] - trace[idx]["time:timestamp"]
                        t = interval_size * math.ceil((time.total_seconds() / 60 / 60) / interval_size)
                        l.append(e["concept:name"] + "_" + str(t))
                        open_set.pop(idx)
                        found = True
                        break
                if not found:
                    if ide == 0:
                        l.append(e["concept:name"]+"_"+str(interval_size))
                    else:
                        time = trace[ide]["time:timestamp"] - trace[ide-1]["time:timestamp"]
                        t = interval_size * math.ceil((time.total_seconds() / 60 / 60) / interval_size)
                        l.append(e["concept:name"] + "_" + str(t))
    else:
        if len(trace) == 1:
            return [trace[0]["concept:name"] + "_0"]
        t = sorted(trace, key=lambda d: d['time:timestamp'])
        for idx,e in enumerate(trace):
            if idx == 0:
                l.append(e["concept:name"]+"_0")
            else:
                time = trace[idx]["time:timestamp"] - trace[idx-1]["time:timestamp"]
                t = interval_size * math.ceil((time.total_seconds()/60/60)/interval_size)
                l.append(e["concept:name"]+"_"+str(t))
    #print(l, len(trace))
    return l


def retrieve_timed_activity_exponential(trace):
    l = []
    if len(trace) == 1:
        return [trace[0]["concept:name"] + "_2"]
    if "lifecycle:transition" in trace[0]:
        open_set = []
        for ide,e in enumerate(trace):
            if "start" in e["lifecycle:transition"].lower():
                open_set.append(e)
            elif "complete" in e["lifecycle:transition"].lower():
                found = False
                for idx, x in enumerate(open_set):
                    if x["concept:name"] == e["concept:name"]:
                        time = trace[ide]["time:timestamp"] - trace[idx]["time:timestamp"]
                        t = time.total_seconds() / 60 / 60
                        if t < 1:
                            l.append(e["concept:name"] + "_2")
                        else:
                            l.append(e["concept:name"] + "_" + str(2 * math.ceil(math.log(t, 2))))
                        open_set.pop(idx)
                        found = True
                        break
                if not found:
                    if ide == 0:
                        l.append(e["concept:name"]+"_2")
                    else:
                        time = trace[ide]["time:timestamp"] - trace[ide-1]["time:timestamp"]
                        t = time.total_seconds() / 60 / 60
                        if t < 1:
                            l.append(e["concept:name"] + "_2")
                        else:
                            l.append(e["concept:name"] + "_" + str(2 * math.ceil(math.log(t, 2))))
    else:
        t = sorted(trace, key=lambda d: d['time:timestamp'])
        for idx, e in enumerate(t):
            if idx == 0:
                l.append(e["concept:name"] + "_0")
            else:
                time = trace[idx]["time:timestamp"] - trace[idx-1]["time:timestamp"]
                t = time.total_seconds() / 60/60
                if t < 1:
                    l.append(e["concept:name"] + "_2")
                else:
                    l.append(e["concept:name"] + "_" + str(2 * math.ceil(math.log(t,2))))
    return l


#log = pm4py.read_xes("logs/Sepsis_Cases_-_Event_Log.xes", return_legacy_log_object=True)
#for x in log:
#    retrieve_timed_activity_exponential(x)