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

def plot_gc(events, title = "benchmark", fignum=0):
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
                time_major_gc_end.append(time_threshold[-1])
                threshold.append(event["new-threshold"])
            elif "FINALIZING" not in event["text"] and "SCANNING" in event["text"]:
                time_major_gc_start.append(time_gc_start[-1])


    t_threshold = np.linspace(0, time_memory[-1], len(time_memory)*50)
    y_threshold = heap_limit(t_threshold)
    
    plt.figure(fignum, figsize=(10, 8))
    plt.clf()
    plt.subplot(211)
    for i in range(len(time_major_gc_start)):
        plt.axvspan(time_major_gc_start[i], time_major_gc_end[i], alpha=0.3, color='red')
    plt.plot(time_memory,memory, 'b-', label="heap usage at minor gc")
    plt.plot(t_threshold, y_threshold, 'm', label="heap limit")
    plt.grid()
    ax = plt.gca()
    ax.yaxis.set_major_formatter(EngFormatter("B"))
    ax.xaxis.set_major_formatter(EngFormatter("s"))
    plt.ylabel("memory")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_membalancer_heap_rule(events, mem_balancer, fignum):
    time_threshold = [0]
    heap_limit = [events[4]["new-threshold"]]
    nursery_size = events[4]["nursery-size"]
    mem_balancer.nursery_size = nursery_size
    mem_balancer.s_m_smoothed = 1
    mem_balancer.s_t_smoothed = 1
    # mem_balancer.L_smoothed = events[5]["memory-before-collect"]
    total_memory_used_last_minor = 0
    for event in events[1:]:
        if event["task"] == "gc-minor":
            # total_memory_used = event["memory-before-collect"]
            total_memory_used = event["memory-after-collect"]
            g_m_next = total_memory_used - total_memory_used_last_minor
            if g_m_next >= 0:
                # memory allocated since last heartbeat
                g_m = g_m_next
                print(f"g_m: {g_m}")
            else:
                # g_m = 0
                print(f"g_m_next: {g_m_next}")
            total_memory_used_last_minor = event["memory-after-collect"]
            g_t = event["time-last-minor-gc"]  # time since last heartbeat
            mem_balancer.on_heartbeat(g_m, g_t)
            time_threshold.append(event["time-start"])
            heap_limit.append(mem_balancer.compute_M())
            print(f"heap_limit: {mem_balancer.heap_limit}, E: {
                  mem_balancer.E}")
        elif event["task"] == "gc-collect-step":
            if "SCANNING" in event["text"]:
                print("major collect start")
                time_taken = event["time-taken"]
            else:
                time_taken += event["time-taken"]
        elif event["task"] == "gc-collect-done":
            time_threshold.append(event["time-start"])
            bytes_collected = event["bytes-collected"]
            time_taken += event["time-taken"]
            live_memory = event["memory-after-collect"]
            mem_balancer.on_gc(bytes_collected, time_taken, live_memory)
            # mem_balancer.on_heartbeat()
            print(f"s_m_smoothed: {mem_balancer.s_m_smoothed}, s_t_smoothed: {mem_balancer.s_t_smoothed}, g_m_smoothed: {
                  mem_balancer.g_m_smoothed}, g_t_smoothed: {mem_balancer.g_t_smoothed}")
            heap_limit.append(mem_balancer.compute_M())
            print(f"heap_limit: {mem_balancer.heap_limit}, E: {
                  mem_balancer.E}")
            print("major collect done")

    t = np.linspace(0, events[-1]["time-start"], len(time_threshold)*50)[1:]
    index = np.searchsorted(time_threshold, t, 'right')
    y = np.array(heap_limit)[index-1]
    plt.figure(fignum)
    plt.subplot(211)
    plt.plot(t, y, 'k', label="MemBalancer heap rule")
    plt.legend()
