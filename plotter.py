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


def plot_full_gc_info(log_data : LogData, title="benchmark", fig_num=0) -> None:
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
        num=fig_num,
        sharex=True
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


def plot_benchmark_info(benchmark: list[LogData], tuning_factors: list[float], fig_num: int, title="benchmark data") -> None:
    total_major_gc_time_per_param, total_minor_gc_time_per_param = [], []
    num_minor_gc_per_param, num_major_gc_per_param = [], []
    avg_max_heap_use_per_param, total_heap_use_per_param = [], []
    runtime_per_param, wallclock_time_per_param = [], []
    max_rss_per_param = []

    has_time_info = hasattr(benchmark[0][0], 'user_time')

    for bench in benchmark:
        total_major_gc_time, total_minor_gc_time = [], []
        num_major_gc, num_minor_gc = [], []
        max_heap_use = []
        runtimes = []
        wallclock_times = []
        max_rss_per_run = []
        for prog_run in bench:
            major_gc_time = 0
            for i in range(len(prog_run.time_major_gc_start)):
                major_gc_time += prog_run.time_gc_collect_end[i] - \
                    prog_run.time_gc_collect_start[i]
            total_major_gc_time.append(major_gc_time)
            total_minor_gc_time.append(prog_run.total_minor_gc_time)
            num_major_gc.append(prog_run.major_gcs)
            num_minor_gc.append(prog_run.minor_gcs)
            max_heap_use.append(max(prog_run.memory))
            runtimes.append(prog_run.gc_events[-1]["time-start"])
            if has_time_info:
                wallclock_times.append(prog_run.user_time)
                max_rss_per_run.append(prog_run.max_rss)

        total_heap_use_per_param.append(max_heap_use)
        avg_max_heap_use_per_param.append(np.average(max_heap_use))
        total_major_gc_time_per_param.append(total_major_gc_time)
        total_minor_gc_time_per_param.append(total_minor_gc_time)
        num_major_gc_per_param.append(num_major_gc)
        num_minor_gc_per_param.append(num_minor_gc)
        runtime_per_param.append(runtimes)
        wallclock_time_per_param.append(wallclock_times)
        max_rss_per_param.append(max_rss_per_run)


    plt.close(fig_num)
    if has_time_info:
        subplots = [
            ['major_gc-tuning', 'num_major_gc-tuning'], 
            ['minor_gc-tuning', 'num_minor_gc-tuning'],
            ['total_heap-tuning', 'rss-tuning'], 
            ['runtime-tuning', 'wallclock-tuning'],
            ['major_gc-total_heap', 'major_gc-total_heap'],
            ['runtime-total_heap', 'runtime-total_heap']
        ]
    else:
        subplots = [
            ['major_gc-tuning', 'num_major_gc-tuning'],
            ['minor_gc-tuning', 'num_minor_gc-tuning'], 
            ['total_heap-tuning', 'total_heap-tuning'],
            ['runtime-tuning', 'runtime-tuning'],
            ['major_gc-total_heap', 'major_gc-total_heap'],
            ['runtime-total_heap', 'runtime-total_heap']
        ]
    fig, ax = plt.subplot_mosaic(subplots, figsize=(15, 15), num=fig_num)
    fig.suptitle(title, fontsize=16, y=0.993)

    for i in ax:
        ax[i].grid()

    ax['major_gc-tuning'].boxplot(total_major_gc_time_per_param, showfliers=True)
    ax['major_gc-tuning'].set_xticklabels(tuning_factors,
                                          rotation=30, ha='right')
    ax['major_gc-tuning'].axes.set_ylabel("total major gc time")
    ax['major_gc-tuning'].axes.set_xlabel("tuning factor (rounded)")
    ax['major_gc-tuning'].yaxis.set_major_formatter(EngFormatter("s"))

    ax['minor_gc-tuning'].boxplot(total_minor_gc_time_per_param,
                                  showfliers=True)
    ax['minor_gc-tuning'].set_xticklabels(tuning_factors,
                                          rotation=30, ha='right')
    ax['minor_gc-tuning'].axes.set_ylabel("total minor gc time")
    ax['minor_gc-tuning'].axes.set_xlabel("tuning factor (rounded)")
    ax['minor_gc-tuning'].yaxis.set_major_formatter(EngFormatter("s"))

    ax['num_major_gc-tuning'].boxplot(num_major_gc_per_param, showfliers=True)
    ax['num_major_gc-tuning'].set_xticklabels(
        tuning_factors, rotation=30, ha='right')
    ax['num_major_gc-tuning'].axes.set_ylabel("number of major gcs")
    ax['num_major_gc-tuning'].axes.set_xlabel("tuning factor (rounded)")

    ax['num_minor_gc-tuning'].boxplot(num_minor_gc_per_param, showfliers=True)
    ax['num_minor_gc-tuning'].set_xticklabels(
        tuning_factors, rotation=30, ha='right')
    ax['num_minor_gc-tuning'].axes.set_ylabel("number of minor gcs")
    ax['num_minor_gc-tuning'].axes.set_xlabel("tuning factor (rounded)")

    ax['total_heap-tuning'].boxplot(total_heap_use_per_param, showfliers=True)
    ax['total_heap-tuning'].set_xticklabels(
        tuning_factors, rotation=30, ha='right')
    ax['total_heap-tuning'].axes.set_xlabel("tuning factor (rounded)")
    ax['total_heap-tuning'].axes.set_ylabel("total heap usage") # (max value from minor debug info)
    ax['total_heap-tuning'].yaxis.set_major_formatter(EngFormatter("B"))

    ax['runtime-tuning'].boxplot(runtime_per_param, showfliers=True)
    ax['runtime-tuning'].set_xticklabels(tuning_factors,
                                         rotation=30, ha='right')
    ax['runtime-tuning'].axes.set_xlabel("tuning factor (rounded)")
    ax['runtime-tuning'].axes.set_ylabel("runtime") # (time of last event since start)
    ax['runtime-tuning'].yaxis.set_major_formatter(EngFormatter("s"))

    if has_time_info:
        ax['wallclock-tuning'].boxplot(wallclock_time_per_param, showfliers=True)
        ax['wallclock-tuning'].set_xticklabels(
            tuning_factors, rotation=30, ha='right')
        ax['wallclock-tuning'].axes.set_xlabel("tuning factor (rounded)")
        ax['wallclock-tuning'].axes.set_ylabel(
            "wall clock time")
        ax['wallclock-tuning'].yaxis.set_major_formatter(EngFormatter("s"))

        ax['rss-tuning'].boxplot(max_rss_per_param, showfliers=True)
        ax['rss-tuning'].set_xticklabels(tuning_factors,
                                         rotation=30, ha='right')
        ax['rss-tuning'].axes.set_xlabel("tuning factor (rounded)")
        ax['rss-tuning'].axes.set_ylabel(
            "max rss")
        ax['rss-tuning'].yaxis.set_major_formatter(EngFormatter("B"))

    colors = plt.cm.tab20b(np.linspace(0, 1, len(total_heap_use_per_param)))
    for i in range(len(total_heap_use_per_param)):
        ax['major_gc-total_heap'].scatter(total_heap_use_per_param[i], total_major_gc_time_per_param[i],
                      color=colors[i], alpha=0.8, s=10)
        ax['major_gc-total_heap'].plot(np.average(total_heap_use_per_param[i]), np.average(total_major_gc_time_per_param[i]),
                      color=colors[i], marker='d', ms=6)
        ax['runtime-total_heap'].scatter(total_heap_use_per_param[i],
                                         runtime_per_param[i], color=colors[i], alpha=0.8, s=10)
        ax['runtime-total_heap'].plot(np.average(total_heap_use_per_param[i]),
                   np.average(runtime_per_param[i]), color=colors[i], marker='d', ms=6)
    ax['major_gc-total_heap'].plot(np.average(total_heap_use_per_param, axis=1),
               np.average(total_major_gc_time_per_param, axis=1), 'kd--', alpha=0.6, ms=4)
    ax['runtime-total_heap'].plot(np.average(total_heap_use_per_param, axis=1),
               np.average(runtime_per_param, axis=1), 'kd--', alpha=0.6, ms=4)
    
    ax['major_gc-total_heap'].yaxis.set_major_formatter(EngFormatter("s"))
    ax['major_gc-total_heap'].xaxis.set_major_formatter(EngFormatter("B"))
    ax['major_gc-total_heap'].axes.set_ylabel("total major gc collection time")
    ax['major_gc-total_heap'].axes.set_xlabel("total heap usage") # (max value of iteration)
    #plt.ylim(bottom=0)
    #plt.xlim(left=0)

    ax['runtime-total_heap'].yaxis.set_major_formatter(EngFormatter("s"))
    ax['runtime-total_heap'].xaxis.set_major_formatter(EngFormatter("B"))
    ax['runtime-total_heap'].axes.set_ylabel("runtime") # (time of last event since start)
    ax['runtime-total_heap'].axes.set_xlabel("total heap usage") # (max value of iteration)

    fig.canvas.header_visible = False
    fig.tight_layout()
    fig.set_facecolor('slategrey')


def plot_pareto(bench1, bench2, fig_num, bench1_label = "bench1", bench2_label = "bench2", title="benchmark data"):
    total_major_gc_time_per_param_1, total_heap_use_per_param_1, runtime_per_param_1 = [], [], []
    total_major_gc_time_per_param_2, total_heap_use_per_param_2, runtime_per_param_2 = [], [], []

    for bench in bench1:
        total_major_gc_time = []
        max_heap_use = []
        runtimes = []
        for prog_run in bench:
            major_gc_time = 0
            for i in range(len(prog_run.time_major_gc_start)):
                major_gc_time += prog_run.time_gc_collect_end[i] - \
                    prog_run.time_gc_collect_start[i]
            total_major_gc_time.append(major_gc_time)
            max_heap_use.append(sum(prog_run.memory)/len(prog_run.memory))
            runtimes.append(prog_run.gc_events[-1]["time-start"])

        total_heap_use_per_param_1.append(max_heap_use)
        total_major_gc_time_per_param_1.append(total_major_gc_time)
        runtime_per_param_1.append(runtimes)
    
    for bench in bench2:
        total_major_gc_time = []
        max_heap_use = []
        runtimes = []
        for prog_run in bench:
            major_gc_time = 0
            for i in range(len(prog_run.time_major_gc_start)):
                major_gc_time += prog_run.time_gc_collect_end[i] - \
                    prog_run.time_gc_collect_start[i]
            total_major_gc_time.append(major_gc_time)
            max_heap_use.append(sum(prog_run.memory)/len(prog_run.memory))
            runtimes.append(prog_run.gc_events[-1]["time-start"])

        total_heap_use_per_param_2.append(max_heap_use)
        total_major_gc_time_per_param_2.append(total_major_gc_time)
        runtime_per_param_2.append(runtimes)


    plt.close(fig_num)
    subplots = [
        ['major_gc-total_heap'],
        ['runtime-total_heap']
        ]
    
    fig, ax = plt.subplot_mosaic(subplots, figsize=(15, 8), num=fig_num)
    fig.suptitle(title, fontsize=16, y=0.993)

    for i in ax:
        ax[i].grid()

    for i in range(len(total_heap_use_per_param_1)):
        ax['major_gc-total_heap'].scatter(total_heap_use_per_param_1[i], total_major_gc_time_per_param_1[i],
                                          color='k', alpha=0.8, s=10)
        #ax['major_gc-total_heap'].plot(np.average(total_heap_use_per_param_1[i]), np.average(total_major_gc_time_per_param_1[i]),
        #                               color='k', marker='d', ms=6)
        ax['runtime-total_heap'].scatter(total_heap_use_per_param_1[i],
                                         runtime_per_param_1[i], color='k', alpha=0.8, s=10)
        #ax['runtime-total_heap'].plot(np.average(total_heap_use_per_param_1[i]),
        #                              np.average(runtime_per_param_1[i]), color='k', marker='d', ms=6)
    ax['major_gc-total_heap'].plot(np.average(total_heap_use_per_param_1, axis=1),
                                   np.average(total_major_gc_time_per_param_1, axis=1), 'kd--', alpha=0.6, ms=4, label=bench1_label)
    ax['runtime-total_heap'].plot(np.average(total_heap_use_per_param_1, axis=1),
                                  np.average(runtime_per_param_1, axis=1), 'kd--', alpha=0.6, ms=4, label=bench1_label)
    
    for i in range(len(total_heap_use_per_param_2)):
        ax['major_gc-total_heap'].scatter(total_heap_use_per_param_2[i], total_major_gc_time_per_param_2[i],
                                          color='b', alpha=0.8, s=10)
        #ax['major_gc-total_heap'].plot(np.average(total_heap_use_per_param_2[i]), np.average(total_major_gc_time_per_param_2[i]),
        #                               color='b', marker='d', ms=6)
        ax['runtime-total_heap'].scatter(total_heap_use_per_param_2[i],
                                         runtime_per_param_2[i], color='b', alpha=0.8, s=10)
        #ax['runtime-total_heap'].plot(np.average(total_heap_use_per_param_2[i]),
        #                              np.average(runtime_per_param_2[i]), color='b', marker='d', ms=6)
    ax['major_gc-total_heap'].plot(np.average(total_heap_use_per_param_2, axis=1),
                                   np.average(total_major_gc_time_per_param_2, axis=1), 'bd--', alpha=0.6, ms=4, label=bench2_label)
    ax['runtime-total_heap'].plot(np.average(total_heap_use_per_param_2, axis=1),
                                  np.average(runtime_per_param_2, axis=1), 'bd--', alpha=0.6, ms=4, label=bench2_label)

    ax['major_gc-total_heap'].yaxis.set_major_formatter(EngFormatter("s"))
    ax['major_gc-total_heap'].xaxis.set_major_formatter(EngFormatter("B"))
    ax['major_gc-total_heap'].axes.set_ylabel("total major gc collection time")
    # (max value of iteration)
    ax['major_gc-total_heap'].axes.set_xlabel("total heap usage")
    ax['major_gc-total_heap'].legend()
    # plt.ylim(bottom=0)
    # plt.xlim(left=0)

    ax['runtime-total_heap'].yaxis.set_major_formatter(EngFormatter("s"))
    ax['runtime-total_heap'].xaxis.set_major_formatter(EngFormatter("B"))
    # (time of last event since start)
    ax['runtime-total_heap'].axes.set_ylabel("runtime")
    # (max value of iteration)
    ax['runtime-total_heap'].axes.set_xlabel("total heap usage")
    ax['runtime-total_heap'].legend()

    fig.canvas.header_visible = False
    fig.tight_layout()
    fig.set_facecolor('slategrey')
