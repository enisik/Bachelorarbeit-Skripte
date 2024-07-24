from logparser import LogData
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.widgets import CheckButtons
import numpy as np

def plot_area(starts, ends, ax, alpha, color):
    for i in range(len(starts)):
        ax.axvspan(starts[i], ends[i], alpha=alpha, color=color)

def plot_gc(events, title="benchmark", fig_num=0):
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
    
    time_threshold.append(time_memory[-1])
    threshold.append(threshold[-1])

    fig = plt.figure(fig_num, figsize=(13, 9))
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

def plot_membalancer_heap_rule(events, mem_balancer, fig_num):
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

    fig = plt.figure(fig_num)
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


def plot_full_gc_info(log_data : LogData, title="benchmark", fig_num=0):
    def callback(label, lines_by_label):
        ln = lines_by_label[label]
        ln.set_visible(not ln.get_visible())
        ln.figure.canvas.draw_idle()
    
    g_m_list = np.array(log_data.g_m_list)
    g_m_smoothed_list = np.array(log_data.g_m_smoothed_list)
    g_t_list = np.array(log_data.g_t_list)
    g_t_smoothed_list = np.array(log_data.g_t_smoothed_list)

    log_data.s_m_list.append(log_data.s_m_list[-1])
    log_data.s_t_list.append(log_data.s_t_list[-1])
    log_data.s_m_smoothed_list.append(log_data.s_m_smoothed_list[-1])
    log_data.s_t_smoothed_list.append(log_data.s_t_smoothed_list[-1])
    log_data.time_on_gc.append(log_data.time_memory[-1])

    s_m_list = np.array(log_data.s_m_list)
    s_m_smoothed_list = np.array(log_data.s_m_smoothed_list)
    s_t_list = np.array(log_data.s_t_list)
    s_t_smoothed_list = np.array(log_data.s_t_smoothed_list)

    plt.close(fig_num)
    #fig, ax = plt.subplots(4, 1, sharex=True, num=fig_num, figsize=(13, 12))
    fig, ax = plt.subplot_mosaic([
        [0, 'leg0'],
        [1, 'leg1'],
        [2, 'leg2']
        #[3, 'leg']
    ],
        width_ratios=[5, 1],
        figsize=(15, 9),
        num=fig_num
        )
    fig.suptitle(title, fontsize=16)

    for i in range(3):
        a = ax[i]
        a.grid()
        a.tick_params(labelbottom=True)
        a.set_facecolor('gainsboro')
        plot_area(log_data.time_gc_collect_start,
                  log_data.time_gc_collect_end, a, alpha=0.4, color='red')
        plot_area(log_data.time_major_gc_start,
                  log_data.time_major_gc_end, a, alpha=0.2, color='blue')
        ax[f'leg{i}'].set_facecolor('lightgoldenrodyellow')
    
    ax0_lines_by_label = dict()
    ax1_lines_by_label = dict()
    ax2_lines_by_label = dict()

    line, = ax[0].plot(log_data.time_memory, log_data.memory,
            'b-', label="heap usage")
    ax0_lines_by_label[line.get_label()] = line
    line, = ax[0].step(log_data.time_threshold, log_data.threshold,
           'r', where='post', label="threshold")
    ax0_lines_by_label[line.get_label()] = line
    line, = ax[0].step(log_data.time_threshold[1:], log_data.membalancer_compute_threshold,
               'k', where='post', label="compute_threshold")
    ax0_lines_by_label[line.get_label()] = line
    ax[0].yaxis.set_major_formatter(EngFormatter("B"))
    ax[0].xaxis.set_major_formatter(EngFormatter("s"))
    #ax[0].legend(loc='best',  ncol=2, fancybox=True)

    line, = ax[0].plot(log_data.time_heartbeat, g_m_list, 'goldenrod', label="$g_m$")
    ax0_lines_by_label[line.get_label()] = line
    line, = ax[0].plot(log_data.time_heartbeat, g_m_smoothed_list, 'darkorange', label="$g_m^{*}$")
    ax0_lines_by_label[line.get_label()] = line
    line, = ax[0].step(log_data.time_on_gc, s_m_list, 'darkgreen', where='post', label="$s_m$")
    ax0_lines_by_label[line.get_label()] = line
    line, = ax[0].step(log_data.time_on_gc, s_m_smoothed_list, 'limegreen',
               where='post', label="$s_m^{*}$")
    ax0_lines_by_label[line.get_label()] = line
    #ax[1].yaxis.set_major_formatter(EngFormatter("B"))
    #ax[1].xaxis.set_major_formatter(EngFormatter("s"))
    #ax[1].legend(loc='best', ncol=2, fancybox=True)

    line, = ax[1].plot(log_data.time_heartbeat, g_t_list,
                       'goldenrod', label="$g_t$")
    ax1_lines_by_label[line.get_label()] = line
    line, = ax[1].plot(log_data.time_heartbeat,
                       g_t_smoothed_list, 'darkorange', label="$g_t^{*}$")
    ax1_lines_by_label[line.get_label()] = line
    line, = ax[1].step(log_data.time_on_gc, s_t_list,
                       'darkgreen', where='post', label="$s_t$")
    ax1_lines_by_label[line.get_label()] = line
    line, = ax[1].step(log_data.time_on_gc, s_t_smoothed_list, 'limegreen',
               where='post', label="$s_t^{*}$")
    ax1_lines_by_label[line.get_label()] = line
    ax[1].yaxis.set_major_formatter(EngFormatter("s"))
    ax[1].xaxis.set_major_formatter(EngFormatter("s"))
    #ax[1].legend(loc='best', ncol=2, fancybox=True)

    line, = ax[2].plot(log_data.time_heartbeat, g_m_list /
                       g_t_list, 'goldenrod', label="$g_m / g_t$")
    ax2_lines_by_label[line.get_label()] = line
    line, = ax[2].plot(log_data.time_heartbeat, g_m_smoothed_list /
                       log_data.g_t_smoothed_list, 'darkorange', label="$g_m^{*} / g_t^{*}$")
    ax2_lines_by_label[line.get_label()] = line
    line, = ax[2].step(log_data.time_on_gc, s_m_list/s_t_list, 'darkgreen',
               where='post', label="$s_m / s_t$")
    ax2_lines_by_label[line.get_label()] = line
    line, = ax[2].step(log_data.time_on_gc, s_m_smoothed_list /
                       log_data.s_t_smoothed_list, 'limegreen', where='post', label="$s_m^{*} / s_t^{*}$")
    ax2_lines_by_label[line.get_label()] = line
    ax[2].yaxis.set_major_formatter(EngFormatter("B/s"))
    ax[2].xaxis.set_major_formatter(EngFormatter("s"))
    #ax[2].legend(loc='best', ncol=2, fancybox=True)

    ax0_line_colors = [l.get_color() for l in ax0_lines_by_label.values()]
    ax0_check = CheckButtons(
        ax= ax['leg0'],
        labels=ax0_lines_by_label.keys(),
        actives=[l.get_visible() for l in ax0_lines_by_label.values()],
        label_props={'color': ax0_line_colors, 'size': 
                ['large'] * len(ax0_lines_by_label)},
        frame_props={'edgecolor': ax0_line_colors},
        check_props={'facecolor': ax0_line_colors},
    )

    ax1_line_colors = [l.get_color() for l in ax1_lines_by_label.values()]
    ax1_check = CheckButtons(
        ax=ax['leg1'],
        labels=ax1_lines_by_label.keys(),
        actives=[l.get_visible() for l in ax1_lines_by_label.values()],
        label_props={'color': ax1_line_colors, 'size': 
            ['x-large'] * len(ax1_lines_by_label)},
        frame_props={'edgecolor': ax1_line_colors},
        check_props={'facecolor': ax1_line_colors},
    )

    ax2_line_colors = [l.get_color() for l in ax2_lines_by_label.values()]
    ax2_check = CheckButtons(
        ax=ax['leg2'],
        labels=ax2_lines_by_label.keys(),
        actives=[l.get_visible() for l in ax2_lines_by_label.values()],
        label_props={'color': ax2_line_colors, 'size':
                     ['x-large'] * len(ax2_lines_by_label)}, 
        frame_props={'edgecolor': ax2_line_colors},
        check_props={'facecolor': ax2_line_colors},
    )

    ax0_check.on_clicked(lambda label: callback(label, ax0_lines_by_label))
    ax1_check.on_clicked(lambda label: callback(label, ax1_lines_by_label))
    ax2_check.on_clicked(lambda label: callback(label, ax2_lines_by_label))


    fig.canvas.header_visible = False
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.09, bottom=0.05)
    fig.set_facecolor('slategrey')


def plot_benchmark_info(benchmark: list[LogData], tuning_factors: list[int], fig_num: int, title="benchmark data"):
    total_major_gc_time_per_param = []
    avg_max_heap_use_per_param = []
    total_heap_use_per_param = []
    runtime_per_param = []

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
    
    plt.close(fig_num)
    fig, ax = plt.subplot_mosaic([
        [0],
        [1],
        [2],
        [3],
        [4]
    ],
        figsize=(13, 18),
        num=fig_num
    )
    fig.suptitle(title, fontsize=16)

    ax[0].boxplot(total_major_gc_time_per_param, showfliers=True)
    ax[0].set_xticklabels(tuning_factors)
    ax[0].axes.set_ylabel("total major gc collection time")
    ax[0].axes.set_xlabel("tuning factor (rounded)")
    ax[0].yaxis.set_major_formatter(EngFormatter("s"))

    ax[1].boxplot(total_heap_use_per_param, showfliers=True)
    ax[1].set_xticklabels(tuning_factors)
    ax[1].axes.set_xlabel("tuning factor (rounded)")
    ax[1].axes.set_ylabel("total heap usage (max value from minor debug info)")
    ax[1].yaxis.set_major_formatter(EngFormatter("B"))

    ax[2].boxplot(runtime_per_param, showfliers=True)
    ax[2].set_xticklabels(tuning_factors)
    ax[2].axes.set_xlabel("tuning factor (rounded)")
    ax[2].axes.set_ylabel("runtime (time of last event since start)")
    ax[2].yaxis.set_major_formatter(EngFormatter("s"))

    for i in range(len(total_heap_use_per_param)):
        ax[3].scatter(total_heap_use_per_param[i], total_major_gc_time_per_param[i])
        ax[4].scatter(total_heap_use_per_param[i], runtime_per_param[i])
    
    ax[3].yaxis.set_major_formatter(EngFormatter("s"))
    ax[3].xaxis.set_major_formatter(EngFormatter("B"))
    ax[3].axes.set_ylabel("total major gc collection time")
    ax[3].axes.set_xlabel("total heap usage (max value of iteration)")
    #plt.ylim(bottom=0)
    #plt.xlim(left=0)

    ax[4].yaxis.set_major_formatter(EngFormatter("s"))
    ax[4].xaxis.set_major_formatter(EngFormatter("B"))
    ax[4].axes.set_ylabel("runtime (time of last event since start)")
    ax[4].axes.set_xlabel("total heap usage (max value of iteration)")

    fig.canvas.header_visible = False
    fig.tight_layout()
