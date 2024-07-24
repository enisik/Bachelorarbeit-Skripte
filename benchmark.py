import argparse
import datetime
import subprocess
from tqdm import tqdm
import os

# PYPYLOG=gc:logs/bench-i ./pypy-c ~/Desktop/pypy/rpython/translator/goal/gcbench.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--it', type=int, required=True, help='Number of iterations', dest="iterations")
    arguments = parser.parse_args()
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os_env = dict(os.environ)
    file_path = os.path.expanduser(
        "~/Desktop/pypy/rpython/translator/goal/gcbench.py")
    program_path = os.path.expanduser(
        "~/Desktop/pypy/pypy/goal/pypy-c"
    )
    for tuning_factor in [10, 100, 1_000, 10_000, 100_000]:
        dest_path = f"./logs/{now}/{tuning_factor}"
        os.makedirs(dest_path)
        env = os_env | {"PYPYLOG": f"gc:{dest_path}/bench-%d", "PYPY_GC_MEMBALANCER_TUNING": str(tuning_factor)}
        for i in tqdm(range(arguments.iterations), ncols=100):
            subprocess.run(
                [program_path, file_path], env=env, capture_output=True)
        
