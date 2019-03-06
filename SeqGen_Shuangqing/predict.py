import os
import subprocess
import sys

# import torch

NUM_GPUS = 8
NUM_PROCESS_PER_GPU = 1

assert "--gpus" not in sys.argv
print(f"Using {NUM_GPUS} devices.")
print(f"Running {NUM_PROCESS_PER_GPU} per gpu.")

expname_idx = sys.argv.index("--expname")
for gpu_id in range(NUM_GPUS):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    for pid in range(NUM_PROCESS_PER_GPU):
        subprocess.Popen(
            ["python", "train.py", "--gpus", "0"]
            + sys.argv[1:]
            + ["--expname", f"{sys.argv[expname_idx+1]}/gpu-{gpu_id}-{pid}"],
            env=env,
        )
