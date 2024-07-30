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
    parser.add_argument('--dest', type=str, required=True,
                        help='destination', dest="dest_path")
    arguments = parser.parse_args()
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os_env = dict(os.environ)
    file_path = os.path.expanduser(arguments.file_path)
    compiler_path = os.path.expanduser(arguments.compiler_path)
    dest_path_root = os.path.expanduser(arguments.dest_path)
    for tuning_factor in 10 ** np.linspace(2, 5, 20):
        print(
            f"===============\t\t tuning factor: {tuning_factor:<20} \t\t===============")
        dest_path = f"{dest_path_root}/{now}/{tuning_factor}"
        os.makedirs(dest_path)
        env = os_env | {"PYPYLOG": f"gc:{dest_path}/bench-%d", "PYPY_GC_MEMBALANCER_TUNING": str(tuning_factor)}
        for i in tqdm(range(arguments.iterations), ncols=100):
            subprocess.run(
                [compiler_path, file_path], env=env, capture_output=True)
