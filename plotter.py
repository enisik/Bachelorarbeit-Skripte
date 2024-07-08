import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter


def plot_gc_hex(events, title="benchmark", fignum=0):
    def heap_limit(instruction_number):
        index = np.searchsorted(time_threshold, instruction_number, 'right')
        return np.array(threshold)[index-1]

    minor = [event for event in events if event["task"] == "gc-minor"]
    memory_used = np.array([event["memory-after-collect"] for event in minor])
    
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

def plot_gc(events, title="benchmark", fignum=0):
    def heap_limit(time):
        index = np.searchsorted(time_threshold, time, 'right')
        return np.array(threshold)[index - 1]


    gc_events = [event for event in events if event["task"] == "gc-minor" or "gc-collect" in event["task"]]
    
    time_memory, memory = [], []
    time_threshold, threshold = [0], [events[4]["new-threshold"]]
    time_gc_start, time_gc_end = [], []
    time_major_gc_start, time_major_gc_end = [], []

    for event in gc_events:
        if event["task"] == "gc-minor":
            time_memory.append(event["time-start"])
            memory.append(event["memory-before-collect"])
            time_memory.append(time_memory[-1] + event["time-taken"])
            memory.append(event["memory-after-collect"])
        else:
            time_gc_start.append(event["time-start"])
            time_gc_end.append(time_gc_start[-1] + event["time-taken"])
            if "gc-collect-done" == event["task"]:
                time_threshold.append(time_gc_end[-1])
                threshold.append(event["new-threshold"])
            elif "FINALIZING" in event["text"]:
                time_major_gc_end.append(time_gc_end[-1])
            elif "SCANNING" in event["text"]:
                time_major_gc_start.append(time_gc_start[-1])


    t_threshold = np.linspace(0, time_memory[-1], len(time_memory)*50)
    y_threshold = heap_limit(t_threshold)
    
    fig = plt.figure(fignum, figsize=(15, 10))
    plt.clf()
    fig.suptitle(title, fontsize=16)
    ax = plt.subplot(311)
    for i in range(len(time_gc_start)):
        ax.axvspan(
            time_gc_start[i], time_gc_end[i], alpha=0.4, color='red')

    for i in range(len(time_major_gc_start)):
        ax.axvspan(time_major_gc_start[i], time_major_gc_end[i], alpha=0.2, color='blue')
    ax.plot(time_memory,memory, 'b-', label="heap usage at minor gc")
    ax.plot(t_threshold, y_threshold, 'r', label="heap limit")
    ax.grid()
    ax.yaxis.set_major_formatter(EngFormatter("B"))
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.legend(loc='best',  ncol=2, #bbox_to_anchor=(0.058, 1.05),
              fancybox=True)
    #plt.xlabel("time")
    #plt.ylabel("memory")
    #plt.title(title) 
    #plt.show()


def plot_membalancer_heap_rule(events, mem_balancer, fignum):
    time_threshold = []
    heap_limit = []
    nursery_size = events[4]["nursery-size"]
    mem_balancer.nursery_size = nursery_size
    total_memory_used_last_minor = 0
    time_heartbeat = []
    time_major = []
    g_m_list, g_t_list = [], []
    s_m_list, s_t_list = [], []
    for event in events[1:]:
        if event["task"] == "gc-minor":
            time_heartbeat.append(event["time-start"])
            # total_memory_used = event["memory-before-collect"]
            total_memory_used = event["memory-after-collect"]
            g_m_next = total_memory_used - total_memory_used_last_minor
            g_m_list.append(g_m_next)
            if g_m_next >= 0:
                # memory allocated since last heartbeat
                g_m = g_m_next
                #print(f"g_m: {g_m}")
            else:
                pass
                # g_m = 0
                #print(f"g_m_next: {g_m_next}")
            total_memory_used_last_minor = event["memory-after-collect"]
            g_t = event["time-last-minor-gc"]  # time since last heartbeat
            g_t_list.append(g_t)
            mem_balancer.on_heartbeat(g_m, g_t)
            time_threshold.append(event["time-start"])
            heap_limit.append(mem_balancer.compute_M())
        elif event["task"] == "gc-collect-step":
            if "SCANNING" in event["text"]:
                time_taken = event["time-taken"]
            else:
                time_taken += event["time-taken"]
        elif event["task"] == "gc-collect-done":
            time_major.append(event["time-start"])
            #time_heartbeat.append(time_major[-1])
            time_threshold.append(event["time-start"]+ event["time-taken"])
            bytes_collected = event["bytes-collected"]
            s_m_list.append(bytes_collected)
            time_taken += event["time-taken"]
            s_t_list.append(time_taken)
            live_memory = event["memory-after-collect"]
            mem_balancer.on_gc(bytes_collected, time_taken, live_memory)
            #g_t = time_heartbeat[-2] - time_heartbeat[-1]
            # mem_balancer.on_heartbeat()
            heap_limit.append(mem_balancer.compute_M())

    t = np.linspace(0, events[-1]["time-start"], len(time_threshold)*50)[1:]
    index = np.searchsorted(time_threshold, t, 'right')
    y = np.array(heap_limit)[index-1]
    plt.figure(fignum)
    ax = plt.subplot(311)
    ax.plot(t, y, 'k', label="MemBalancer heap rule")
    ax.legend(loc='best', ncol=2, #bbox_to_anchor=(0.058, 1.05),
              fancybox=True)
    
    ax = plt.subplot(312)
    ax.grid()
    ax.plot(time_heartbeat, g_m_list, label="$g_m$")
    ax.plot(time_major, s_m_list, label="$s_m$")
    ax.yaxis.set_major_formatter(EngFormatter("B"))
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.legend(loc='best', ncol=2,  # bbox_to_anchor=(0.058, 1.05),
              fancybox=True)
    #plt.ylabel("memory")
    
    ax = plt.subplot(313)
    ax.grid()
    g_m_list = np.array(g_m_list)
    g_t_list = np.array(g_t_list)
    s_m_list = np.array(s_m_list)
    s_t_list = np.array(s_t_list)
    ax.plot(time_heartbeat, g_m_list/g_t_list, label="$g_m / g_t$")
    ax.plot(time_major, s_m_list/s_t_list, label="$s_m / s_t$")
    ax.yaxis.set_major_formatter(EngFormatter("B/s"))
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.legend(loc='best', ncol=2,  # bbox_to_anchor=(0.058, 1.05),
              fancybox=True)
    #plt.ylabel("memory")
