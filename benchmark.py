import argparse
import datetime
import numpy as np
import subprocess
from tqdm import tqdm
import os

# PYPYLOG=gc:logs/bench-i ./pypy-c ~/Desktop/pypy/rpython/translator/goal/gcbench.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--it', type=int, default=10, help='Number of iterations', dest="iterations")
    parser.add_argument('--c', type=str, required=True,
                        help='compiler/interpreter', dest="compiler_path")
    parser.add_argument('--t', type=str, required=True,
                        help='script to run', dest="file_path")
    parser.add_argument('--source', type=str, required=False,
                        help='source argument for gctranslate', dest="opt_source")
    parser.add_argument('--dest', type=str, required=True,
                        help='destination', dest="dest_path")
    parser.add_argument('--no_mem_balancer', action=argparse.BooleanOptionalAction,
                        dest="no_mem_bal")
    arguments = parser.parse_args()
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os_env = dict(os.environ)
    file_path = os.path.expanduser(arguments.file_path)
    compiler_path = os.path.expanduser(arguments.compiler_path)
    dest_path_root = os.path.expanduser(arguments.dest_path)
    args = [compiler_path, file_path]
    if arguments.no_mem_bal:
        env_tunings = [{"PYPY_GC_MAJOR_COLLECT": str(
            tuning_factor)} for tuning_factor in [1.4, 1.8, 2.4, 3, 3.7, 4.4, 5.2, 6]]
        key = "PYPY_GC_MAJOR_COLLECT"
        gc_name = "MiniMark"
    else:
        env_tunings = [{"PYPY_GC_MEMBALANCER_TUNING": str(
            tuning_factor)} for tuning_factor in 10 ** np.linspace(2, 5, 20)]
        key = "PYPY_GC_MEMBALANCER_TUNING"
        gc_name = "MemBalancer"
    if arguments.opt_source is not None:
        args.append("--source")
        args.append(arguments.opt_source)
    for env_tuning in env_tunings:
        print(
            f"===============\t\t {key}:{env_tuning[key]:<20} \t\t===============")
        dest_path = f"{dest_path_root}/{now}/{gc_name}/{env_tuning[key]}"
        os.makedirs(dest_path)
        env = os_env | env_tuning
        for i in tqdm(range(arguments.iterations), ncols=100):
            env["PYPYLOG"] = f"gc:{dest_path}/{i}"
            subprocess.run(args, env=env, capture_output=True)
