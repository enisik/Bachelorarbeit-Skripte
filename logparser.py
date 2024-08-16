import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd

class LogData:

    def __init__(self, path: str, mem_balancer: bool=True):
        self.time_memory, self.memory = [], []
        self.time_threshold, self.threshold = [], []
        self.membalancer_compute_threshold, self.membalancer_limit = [], []
        self.time_gc_collect_start, self.time_gc_collect_end = [], []
        self.time_major_gc_start, self.time_major_gc_end = [], []
        self.time_heartbeat = []
        self.g_m_list, self.g_m_smoothed_list, self.g_t_list, self.g_t_smoothed_list = [], [], [], []
        self.time_on_gc = []
        self.s_m_list, self.s_m_smoothed_list, self.s_t_list, self.s_t_smoothed_list = [], [], [], []
        events = get_events_from(path)
        if "user-time" in events[-1].keys():
            self.user_time = events[-1]["user-time"]
            self.max_rss = events[-1]["rss"]
        self.gc_events = [event for event in events if event["task"]
                     == "gc-minor" or "gc-collect" in event["task"]]
        self.minor_gcs = 0
        self.major_gcs = 0
        
        if mem_balancer:
            for event in self.gc_events:
                if event["task"] == "gc-minor":
                    self.minor_gcs += 1
                    self.time_memory.append(event["time-start"])
                    self.memory.append(event["memory-before-collect"])
                    self.time_memory.append(self.time_memory[-1] + event["time-taken"])
                    self.memory.append(event["memory-after-collect"])
                    self.time_heartbeat.append((self.time_memory[-2] + self.time_memory[-1])/2)
                    self.g_m_list.append(event["g_m"])
                    self.g_m_smoothed_list.append(event["g_m_smoothed"])
                    self.g_t_list.append(event["g_t"])
                    self.g_t_smoothed_list.append(event["g_t_smoothed"])
                    if "new-threshold" in event:
                        self.time_threshold.append(self.time_memory[-1])
                        self.threshold.append(event["new-threshold"])
                        self.membalancer_compute_threshold.append(
                            event["membalancer-compute_threshold"])
                else:
                    self.time_gc_collect_start.append(event["time-start"])
                    self.time_gc_collect_end.append(
                        self.time_gc_collect_start[-1] + event["time-taken"])
                    if "gc-collect-done" == event["task"]:
                        self.major_gcs += 1
                        time = self.time_gc_collect_end[-1]
                        self.time_threshold.append(time)
                        self.threshold.append(event["new-threshold"])
                        self.membalancer_compute_threshold.append(
                            event["membalancer-compute_threshold"])

                        self.time_on_gc.append(time)
                        self.s_m_list.append(event["s_m"])
                        self.s_m_smoothed_list.append(event["s_m_smoothed"])
                        self.s_t_list.append(event["s_t"])
                        self.s_t_smoothed_list.append(event["s_t_smoothed"])

                    elif "FINALIZING" in event["text"]:
                        self.time_major_gc_end.append(self.time_gc_collect_end[-1])
                    elif "SCANNING" in event["text"]:
                        self.time_major_gc_start.append(self.time_gc_collect_start[-1])

            self.membalancer_compute_threshold.append(
                self.membalancer_compute_threshold[-1])
        else:
            for event in self.gc_events:
                if event["task"] == "gc-minor":
                    self.minor_gcs += 1
                    self.time_memory.append(event["time-start"])
                    self.memory.append(event["memory-before-collect"])
                    self.time_memory.append(
                        self.time_memory[-1] + event["time-taken"])
                    self.memory.append(event["memory-after-collect"])
                    if "new-threshold" in event:
                        self.time_threshold.append(self.time_memory[-1])
                        self.threshold.append(event["new-threshold"])
                else:
                    self.time_gc_collect_start.append(event["time-start"])
                    self.time_gc_collect_end.append(
                        self.time_gc_collect_start[-1] + event["time-taken"])
                    if "gc-collect-done" == event["task"]:
                        self.major_gcs += 1
                        time = self.time_gc_collect_end[-1]
                        self.time_threshold.append(time)
                        self.threshold.append(event["new-threshold"])

                    elif "FINALIZING" in event["text"]:
                        self.time_major_gc_end.append(
                            self.time_gc_collect_end[-1])
                    elif "SCANNING" in event["text"]:
                        self.time_major_gc_start.append(
                            self.time_gc_collect_start[-1])

        self.time_threshold.insert(0,0)
        self.threshold.insert(0, self.threshold[0])
        self.time_threshold.append(self.time_memory[-1])
        self.threshold.append(self.threshold[-1])

type Benchmark = list[list[LogData]]

def get_events_from(path: str) -> list[dict]:
    # 
    # https://stackoverflow.com/questions/30627810/how-to-parse-this-custom-log-file-in-python
    #
    import re

    start = re.compile(r'\[[0-9a-f]+\] \{')
    end = re.compile(r'\[[0-9a-f]+\] [a-z-]+}')

    def generateDicts(log_fh):
        currentDict = {}
        for line in log_fh:
            # ignore gc-minor-walkroots entries
            if "gc-minor-walkroots" in line: 
                continue

            if start.match(line):
                splitted_line = line.split("{")
                if splitted_line[1][:-1] == "gc-collect-done":
                    currentDict["task"] = "gc-collect-done"
                else:
                    currentDict = {"start-stamp": splitted_line[0][1:-2], "task": splitted_line[1][:-1], "text": ""}
            
            elif end.match(line) and "gc-collect-done" not in line:
                splitted_line = line.split("]")
                currentDict["end-stamp"] = splitted_line[0][1:]
                yield currentDict
            
            elif "time since program start" in line:
                splitted_line = line.split(":")
                currentDict["time-start"] = float(splitted_line[1])

            elif "time since end of last minor GC" in line: 
                splitted_line = line.split(":")
                currentDict["time-last-minor-gc"] = float(splitted_line[1])

            elif "memory used before collect" in line:
                splitted_line = line.split(":")
                currentDict["memory-before-collect"] = float(splitted_line[1])

            elif "total memory used" in line:
                splitted_line = line.split(":")
                currentDict["memory-after-collect"] = int(splitted_line[1])

            elif "time taken" in line:
                splitted_line = line.split(":")
                if currentDict.get("time-taken") is not None:
                    currentDict["time-taken"] += float(splitted_line[1])
                else:
                    currentDict["time-taken"] = float(splitted_line[1])

            elif "major collection threshold" in line:
                splitted_line = line.split(":")
                currentDict["new-threshold"] = float(splitted_line[1])

            elif "freed in this major collection" in line:
                splitted_line = line.split(":")
                currentDict["bytes-collected"] = float(splitted_line[1])

            elif "nursery size" in line:
                splitted_line = line.split(":")
                currentDict["nursery-size"] = int(splitted_line[1])
            
            elif "total size of surviving objects" in line:
                splitted_line = line.split(":")
                currentDict["surviving-objects"] = int(splitted_line[1])
            
            elif "membalancer on_gc L" in line:
                splitted_line = line.split(":")
                currentDict["L"] = float(splitted_line[1])
            
            elif "membalancer on_gc s_m_smoothed" in line:
                splitted_line = line.split(":")
                currentDict["s_m_smoothed"] = float(splitted_line[1])

            elif "membalancer on_gc s_t_smoothed" in line:
                splitted_line = line.split(":")
                currentDict["s_t_smoothed"] = float(splitted_line[1])

            elif "membalancer on_gc s_m" in line:
                splitted_line = line.split(":")
                currentDict["s_m"] = float(splitted_line[1])

            elif "membalancer on_gc s_t" in line:
                splitted_line = line.split(":")
                currentDict["s_t"] = float(splitted_line[1])

            elif "membalancer on_heartbeat g_m_smoothed" in line:
                splitted_line = line.split(":")
                currentDict["g_m_smoothed"] = float(splitted_line[1])

            elif "membalancer on_heartbeat g_t_smoothed" in line:
                splitted_line = line.split(":")
                currentDict["g_t_smoothed"] = float(splitted_line[1])

            elif "membalancer on_heartbeat g_m" in line:
                splitted_line = line.split(":")
                currentDict["g_m"] = float(splitted_line[1])

            elif "membalancer on_heartbeat g_t" in line:
                splitted_line = line.split(":")
                currentDict["g_t"] = float(splitted_line[1])

            elif "membalancer compute_threshold" in line:
                splitted_line = line.split(":")
                currentDict["membalancer-compute_threshold"] = float(splitted_line[1])

            elif "User time" in line:
                splitted_line = line.split(":")
                currentDict = {"user-time": float(splitted_line[-1]), "text": "", "task": ""}

            elif "Maximum resident set" in line:
                splitted_line = line.split(":")
                currentDict["rss"] = float(splitted_line[-1])
                yield currentDict

            else:
                currentDict["text"] += line

    with open(path, "r") as f:
        events = list(generateDicts(f))
    return events

def get_log_data_from_folder(folder: str, mem_balancer: bool=True) -> list[LogData]:
    only_files = [path for file in listdir(folder) if isfile(path := join(folder, file))]
    data_from_logs = [LogData(path, mem_balancer) for path in only_files]
    return data_from_logs


def get_tuning_factor(path: str) -> float:
    return float(path.split('/')[-1])


def get_benchmark(source: str, mem_balancer: bool=True) -> tuple[Benchmark, list[float]]:
    benchmark = []
    only_folder = [path for file in listdir(
        source) if not isfile(path := join(source, file))]
    only_folder.sort(key=get_tuning_factor)
    for folder in only_folder:
        benchmark.append(get_log_data_from_folder(folder, mem_balancer))
    tuning_factors = [round(get_tuning_factor(path)) for path in only_folder]
    return benchmark, tuning_factors

def get_stats_from_log_data(benchmark: Benchmark, tuning_factors: list[int]):
    total_major_gc_time_per_param = []
    avg_max_heap_use_per_param = []
    total_heap_use_per_param = []
    runtime_per_param = []
    avg_runtime_per_param = []
    data_frames = []

    for bench in benchmark:
        total_major_gc_time = []
        max_heap_use = []
        runtimes = []
        for prog_run in bench:
            time = 0
            for i in range(len(prog_run.time_major_gc_start)):
                time += prog_run.time_gc_collect_end[i] - \
                    prog_run.time_gc_collect_start[i]
            total_major_gc_time.append(time)
            max_heap_use.append(max(prog_run.memory))
            runtimes.append(prog_run.gc_events[-1]["time-start"])

        total_heap_use_per_param.append(max_heap_use)
        avg_max_heap_use_per_param.append(np.average(max_heap_use))
        total_major_gc_time_per_param.append(total_major_gc_time)
        runtime_per_param.append(runtimes)
        avg_runtime_per_param.append(np.average(runtimes))
        data = np.vstack([max_heap_use, total_major_gc_time, runtimes]).T
        data_frames.append(pd.DataFrame(data, columns=[
            "max heap", "total major gc time", "runtime"]))

    data = np.vstack([avg_max_heap_use_per_param, avg_runtime_per_param]).T
    data_frame = pd.DataFrame(data,
                     index=tuning_factors, columns=["avg max heap", "avg runtimes"])

    #data_frame.index.name = "tuning factor"
    return data_frames, data_frame

if __name__ == "__main__":
    events = get_events_from("logs/gcbench")
    for item in events:
        print(item)