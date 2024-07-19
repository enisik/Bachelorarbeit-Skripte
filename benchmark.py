import subprocess
import os
import datetime
from tqdm import tqdm

# PYPYLOG=gc:logs/bench-i ./pypy-c ~/Desktop/pypy/rpython/translator/goal/gcbench.py

if __name__ == "__main__":
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dest_path = f"./logs/{now}"
    os_env = dict(os.environ)
    os.makedirs(dest_path)
    file_path = os.path.expanduser(
        "~/Desktop/pypy/rpython/translator/goal/gcbench.py")
    for i in tqdm(range(30), ncols=100):
        subprocess.run(
            ["./pypy-c", file_path], env=os_env | {"PYPYLOG": f"gc:{dest_path}/bench-{i}"}, capture_output=True)
        
