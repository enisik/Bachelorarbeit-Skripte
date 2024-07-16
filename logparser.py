

def get_events_from(path):
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
                    currentDict = {"start": splitted_line[0][1:-2], "task": splitted_line[1][:-1], "text": ""}
            
            elif end.match(line) and "gc-collect-done" not in line:
                splitted_line = line.split("]")
                currentDict["end"] = splitted_line[0][1:]
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

            elif "membalancer on_gc s_m" in line:
                splitted_line = line.split(":")
                currentDict["s_m"] = float(splitted_line[1])

            elif "membalancer on_gc s_t" in line:
                splitted_line = line.split(":")
                currentDict["s_t"] = float(splitted_line[1])

            elif "membalancer on_gc s_m_smoothed" in line:
                splitted_line = line.split(":")
                currentDict["s_m_smoothed"] = float(splitted_line[1])

            elif "membalancer on_gc s_t_smoothed" in line:
                splitted_line = line.split(":")
                currentDict["s_t_smoothed"] = float(splitted_line[1])

            elif "membalancer on_heartbeat g_m" in line:
                splitted_line = line.split(":")
                currentDict["g_m"] = float(splitted_line[1])

            elif "membalancer on_heartbeat g_t" in line:
                splitted_line = line.split(":")
                currentDict["g_t"] = float(splitted_line[1])

            elif "membalancer on_heartbeat g_m_smoothed" in line:
                splitted_line = line.split(":")
                currentDict["g_m_smoothed"] = float(splitted_line[1])

            elif "membalancer on_heartbeat g_t_smoothed" in line:
                splitted_line = line.split(":")
                currentDict["g_t_smoothed"] = float(splitted_line[1])
            
            elif "membalancer compute_threshold" in line:
                splitted_line = line.split(":")
                currentDict["membalancer-compute_threshold"] = float(splitted_line[1])

            elif "next major collection threshold" in line:
                splitted_line = line.split(":")
                currentDict["membalancer-limit"] = float(splitted_line[1])

            else:
                currentDict["text"] += line


    with open(path, "r") as f:
        events = list(generateDicts(f))
    return events

if __name__ == "__main__":
    events = get_events_from("logs/gcbench")
    for item in events:
        print(item)