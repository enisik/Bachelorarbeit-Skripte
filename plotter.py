import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter


def plot_gc_hex(events, title="benchmark", fignum=0):
    def heap_limit(instruction_number):
        index = np.searchsorted(time_threshold, instruction_number, 'right')
        return np.array(threshold)[index-1]

    minor = [event for event in events if event["task"] == "gc-minor"]
    memory_used = np.array([event["total-memory-used"] for event in minor])
    
    t_minor = np.array([int(event["start"], 16)  for event in minor], dtype=float)
    start_instruction = t_minor[0]
    t_minor -= start_instruction

    time_threshold = [int(events[4]["start"],16) - start_instruction]
    # hardcoded 4th event is second nursery
    threshold = [events[4]["new-threshold"]]
    begin_major_gc, end_major_gc = [], []
    for event in events:
        if event["task"] == "gc-collect-done":
            time_threshold.append(int(event["start"], 16) - start_instruction)
            threshold.append(event["new-threshold"])
    
        elif "FINALIZING" in event["text"]:
            end_major_gc.append(int(event["end"], 16) - start_instruction)
        # assumption: SCANNING only in start and final event
        elif event["task"] == "gc-collect-step" and "SCANNING" in event["text"]:
            begin_major_gc.append(int(event["start"], 16) - start_instruction)
    
    
    t_threshold = np.linspace(time_threshold[0], t_minor[-1], len(t_minor)*50)
    y_threshold = heap_limit(t_threshold)
    plt.figure(fignum)
    plt.clf()
    plt.plot(t_minor,memory_used, 'b-', label="heap usage at minor gc")
    plt.plot(t_threshold, y_threshold, 'm', label="heap limit")
    for i in range(len(begin_major_gc)-1):
        plt.axvspan(begin_major_gc[i], end_major_gc[i], alpha=0.3, color='red')
    plt.axvspan(begin_major_gc[-1], end_major_gc[-1], alpha=0.3, color='red', label="major gc")
    plt.grid()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(EngFormatter("B"))
    ax.xaxis.set_major_formatter(EngFormatter())
    plt.xlabel("instruction number")
    plt.ylabel("memory (1 KB = 1000 B)")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_gc(events, title = "benchmark", fignum=0):
    def heap_limit(time):
        index = np.searchsorted(time_threshold, time, 'right')
        return np.array(threshold)[index-1]


    gc_events = [event for event in events if event["task"] == "gc-minor" or "gc-collect" in event["task"]]
    time_last_minor  = [event.get("time-last-minor-gc") for event in gc_events]
    memory_used = [event.get("total-memory-used") for event in gc_events]
    acc = 0
    time = [0]
    for time_dist in time_last_minor[1:]:
        if time_dist is not None:  
            acc += time_dist
            time.append(acc)
        else:
            time.append(None)
    time_begin_major_gc, time_end_major_gc, time_threshold, threshold = [],[] ,[0] ,[events[4]["new-threshold"]] # hardcoded 4th event is second nursery
    for i in range(len(gc_events)):
        if time[i] is None:
            time[i] = time[i-1] + gc_events[i-1]["time-taken"]
            if gc_events[i]["task"] == "gc-collect-step":
                if "FINALIZING" in gc_events[i]["text"]:
                    time_end_major_gc.append(time[i+1])
                    #time[i] = time[i+1]
                    
                elif "SCANNING" in gc_events[i]["text"]:
                    time_begin_major_gc.append(time[i])
                    #time[i] = time[i-1]  + gc_events[i-1]["time-taken"]
            elif gc_events[i]["task"] == "gc-collect-done" or gc_events[i]["task"] == "gc-set-nursery-size":
                time_threshold.append(time[i])
                threshold.append(gc_events[i]["new-threshold"])

    time_no_none = []
    memory_used_no_none = []
    for i in range(len(memory_used)):
        if memory_used[i] is not None:
            time_no_none.append(time[i])
            memory_used_no_none.append(memory_used[i])

    t_threshold = np.linspace(0, time_no_none[-1], len(time_no_none)*50)
    y_threshold = heap_limit(t_threshold)
    
    plt.figure(fignum)
    plt.clf()
    for i in range(len(time_begin_major_gc)-1):
        plt.axvspan(time_begin_major_gc[i], time_end_major_gc[i], alpha=0.3, color='red')
    plt.axvspan(time_begin_major_gc[-1], time_end_major_gc[-1], alpha=0.3, color='red', label="major gc")
    plt.plot(time_no_none,memory_used_no_none, 'b-', label="heap usage at minor gc")
    plt.plot(t_threshold, y_threshold, 'm', label="heap limit")
    plt.grid()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(EngFormatter("B"))
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    plt.ylabel("memory (1 KB = 1000 B)")
    plt.title(title)
    plt.legend()
    plt.show()