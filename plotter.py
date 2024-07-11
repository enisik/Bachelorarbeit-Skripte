import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter

def plot_area(starts, ends, ax, alpha, color):
    for i in range(len(starts)):
        ax.axvspan(starts[i], ends[i], alpha=alpha, color=color)

def plot_gc(events, title="benchmark", fignum=0):
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
                threshold.append(event["membalancer-compute_threshold"])
            elif "FINALIZING" in event["text"]:
                time_major_gc_end.append(time_gc_end[-1])
            elif "SCANNING" in event["text"]:
                time_major_gc_start.append(time_gc_start[-1])
    
    time_threshold.append(time_memory[-1])
    threshold.append(threshold[-1])

    fig = plt.figure(fignum, figsize=(13, 9))
    plt.clf()
    fig.suptitle(title, fontsize=16)
    fig.canvas.header_visible = False
    ax = plt.subplot(311)
    plot_area(time_gc_start, time_gc_end, ax, alpha=0.4, color='red')
    plot_area(time_major_gc_start, time_major_gc_end,
              ax, alpha=0.2, color='blue')
    ax.plot(time_memory,memory, 'b-', label="heap usage at minor gc")
    ax.step(time_threshold , threshold, 'r', where='post', label="heap limit")
    ax.grid()
    ax.yaxis.set_major_formatter(EngFormatter("B"))
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.legend(loc='best',  ncol=2, fancybox=True)

def plot_membalancer_heap_rule(events, mem_balancer, fignum):
    time_threshold = []
    heap_limit = []
    nursery_size = events[4]["nursery-size"]
    mem_balancer.nursery_size = nursery_size
    time_heartbeat = []
    time_major = []
    g_m_list, g_m_smoothed_list, g_t_list, g_t_smoothed_list = [], [], [], []
    s_m_list, s_m_smoothed_list, s_t_list, s_t_smoothed_list = [], [], [], []
    for event in events[1:]:
        if event["task"] == "gc-minor":
            time_heartbeat.append(event["time-start"])
            total_memory_used_before = event["memory-before-collect"]
            total_memory_used_after = event["memory-after-collect"]
            g_m_next = total_memory_used_after - total_memory_used_before #total_memory_used - total_memory_used_last_minor
            g_m_next = event["surviving-objects"]
            g_m_list.append(g_m_next)
            #if g_m_next >= 0:
                # memory allocated since last heartbeat
            g_m = g_m_next
                #print(f"g_m: {g_m}")
                # g_m = 0
                #print(f"g_m_next: {g_m_next}")
            g_t = event["time-last-minor-gc"]  # time since last heartbeat
            g_t_list.append(g_t)
            mem_balancer.on_heartbeat(g_m, g_t)
            g_m_smoothed_list.append(mem_balancer.g_m_smoothed)
            g_t_smoothed_list.append(mem_balancer.g_t_smoothed)
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
            s_m_smoothed_list.append(mem_balancer.s_m_smoothed)
            s_t_smoothed_list.append(mem_balancer.s_t_smoothed)

    fig = plt.figure(fignum)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15, bottom=0.05)
    fig.set_facecolor('slategrey')
    ax = plt.subplot(311)
    ax.step(time_threshold, heap_limit, 'k', where='post', label="MemBalancer heap rule")
    ax.legend(loc='best', ncol=2, fancybox=True)
    
    ax = plt.subplot(312, sharex=ax)
    ax.grid()
    ax.plot(time_heartbeat, g_m_list, label="$g_m$")
    ax.plot(time_heartbeat, g_m_smoothed_list, label="$g_m^{*}$")
    ax.plot(time_major, s_m_list, label="$s_m$")
    ax.plot(time_major, s_m_smoothed_list, label="$s_m^{*}$")
    ax.yaxis.set_major_formatter(EngFormatter("B"))
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.legend(loc='best', ncol=2, fancybox=True)
    
    ax = plt.subplot(313, sharex=ax)
    ax.grid()
    g_m_list = np.array(g_m_list)
    g_m_smoothed_list = np.array(g_m_smoothed_list)
    g_t_list = np.array(g_t_list)
    s_m_list = np.array(s_m_list)
    s_m_smoothed_list = np.array(s_m_smoothed_list)
    s_t_list = np.array(s_t_list)
    ax.plot(time_heartbeat, g_m_list/g_t_list, label="$g_m / g_t$")
    ax.plot(time_heartbeat, g_m_smoothed_list/g_t_smoothed_list, label="$g_m^{*} / g_t^{*}$")
    ax.plot(time_major, s_m_list/s_t_list, label="$s_m / s_t$")
    ax.plot(time_major, s_m_smoothed_list /
            s_t_smoothed_list, label="$s_m^{*} / s_t^{*}$")
    ax.yaxis.set_major_formatter(EngFormatter("B/s"))
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.legend(loc='best', ncol=2, fancybox=True)


def plot_full_gc_info(events, title="benchmark", fignum=0):
    gc_events = [event for event in events if event["task"]
                 == "gc-minor" or "gc-collect" in event["task"]]

    # get data
    time_memory, memory = [], []
    time_threshold, threshold = [0], [events[4]["new-threshold"]]
    time_gc_step_start, time_gc_step_end = [], []
    time_major_gc_start, time_major_gc_end = [], []
    g_m_list, g_m_smoothed_list, g_t_list, g_t_smoothed_list = [], [], [], []
    s_m_list, s_m_smoothed_list, s_t_list, s_t_smoothed_list = [], [], [], []

    for event in gc_events:
        if event["task"] == "gc-minor":
            time_memory.append(event["time-start"])
            memory.append(event["memory-before-collect"])
            time_memory.append(time_memory[-1] + event["time-taken"])
            memory.append(event["memory-after-collect"])
        else:
            time_gc_step_start.append(event["time-start"])
            time_gc_step_end.append(time_gc_step_start[-1] + event["time-taken"])
            if "gc-collect-done" == event["task"]:
                time_threshold.append(time_gc_step_end[-1])
                threshold.append(event["membalancer-compute_threshold"])
            elif "FINALIZING" in event["text"]:
                time_major_gc_end.append(time_gc_step_end[-1])
            elif "SCANNING" in event["text"]:
                time_major_gc_start.append(time_gc_step_start[-1])


    time_threshold.append(time_memory[-1]), threshold.append(threshold[-1])
    # plot
    fig = plt.figure(fignum, figsize=(13, 9))
    plt.clf()
    fig.suptitle(title, fontsize=16)
    fig.canvas.header_visible = False
    ax = plt.subplot(311)
    plot_area(time_gc_step_start, time_gc_step_end, ax, alpha=0.4, color='red')
    plot_area(time_major_gc_start, time_major_gc_end, ax, alpha=0.2, color='blue')
    ax.plot(time_memory, memory, 'b-', label="heap usage at minor gc")
    ax.step(time_threshold, threshold, 'r', where='post', label="heap limit")
    ax.grid()
    ax.yaxis.set_major_formatter(EngFormatter("B"))
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    ax.legend(loc='best',  ncol=2, fancybox=True)
