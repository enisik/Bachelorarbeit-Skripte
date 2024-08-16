import argparse
import datetime
import numpy as np
import subprocess
from tqdm import tqdm
import os
import sys


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--it', type=int, default=10,
                        help='Number of iterations', dest="iterations")
    parser.add_argument('--c', type=str, required=True,
                        help='compiler/interpreter', dest="compiler_path")
    parser.add_argument('--t', type=str, required=True,
                        help='script to run', dest="file_path")
    parser.add_argument('--dest', type=str, required=True,
                        help='destination', dest="dest_path")
    parser.add_argument('--source', type=str, required=False,
                        help='source argument for gctranslate', dest="opt_source")
    parser.add_argument('--benchmark', type=str, required=False,
                        dest="opt_benchmark")
    parser.add_argument('--n', type=str, required=False,
                        dest="opt_n")
    parser.add_argument('--no_mem_balancer', action=argparse.BooleanOptionalAction,
                        dest="no_mem_bal")

    return parser


def get_tuning_factors(arguments):
    if arguments.no_mem_bal:
        env_tunings = [{"PYPY_GC_MAJOR_COLLECT": str(tuning_factor)} for tuning_factor in 
                       [1.4, 1.8, 2, 2.4, 2.7, 3, 3.4, 3.7, 4, 4.4, 4.7, 5, 5.4, 5.7, 6]]
        key = "PYPY_GC_MAJOR_COLLECT"
        gc_name = "MiniMark"
    else:
        env_tunings = [{"PYPY_GC_MEMBALANCER_TUNING": str(tuning_factor)} for 
                       tuning_factor in 10 ** np.linspace(2, 5, 20)]
        key = "PYPY_GC_MEMBALANCER_TUNING"
        gc_name = "MemBalancer"
    return env_tunings, key, gc_name


def get_args(arguments, file_path, compiler_path):
    args = ["time", "--verbose", compiler_path, file_path]
    if arguments.opt_source is not None:
        args.append("--source")
        args.append(arguments.opt_source)
    if arguments.opt_benchmark is not None:
        args.append("--benchmark")
        args.append(arguments.opt_benchmark)
    if arguments.opt_n is not None:
        args.append("-n")
        args.append(arguments.opt_n)
    return args

if __name__ == "__main__":
    parser = get_argparser()
    arguments = parser.parse_args()
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os_env = dict(os.environ)
    file_path = os.path.expanduser(arguments.file_path)
    compiler_path = os.path.expanduser(arguments.compiler_path)
    dest_path_root = os.path.expanduser(arguments.dest_path)
    env_tunings, key, gc_name = get_tuning_factors(arguments)
    args = get_args(arguments, file_path, compiler_path)
    for env_tuning in env_tunings:
        print(
            f"===============\t\t {key}:{env_tuning[key]:<20} \t\t===============")
        dest_path = f"{dest_path_root}/{gc_name}/{now}/{env_tuning[key]}"
        os.makedirs(dest_path)
        env = os_env | env_tuning
        for i in tqdm(range(arguments.iterations), ncols=100):
            env["PYPYLOG"] = f"gc:{dest_path}/{i}"
            result = subprocess.run(args, env=env, capture_output=True)
            with open(f"{dest_path}/{i}", 'a') as f:
                for substr in result.stderr.decode().split('\n'):
                    if "User time" in substr:
                        print(substr, file=f)
                    elif "Maximum resident set size" in substr:
                        print(substr, file=f)
