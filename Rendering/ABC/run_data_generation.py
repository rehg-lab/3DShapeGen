import os
import argparse
import numpy as np

GPUS = [0,1,2]
n_jobs = 9

start = 35000
end = 52500
total_objects = end - start

obj_range = np.arange(start, end, total_objects//n_jobs)
obj_range[-1] = end

gpu_range = np.array([GPUS] * (n_jobs//len(GPUS))).T.flatten()
dir_path = './job_logs/'

if not os.path.exists(dir_path):
    os.makedirs(dir_path)

cmd = ''
for idx in range(n_jobs):
    start = obj_range[idx]
    end = obj_range[idx+1]
    
    gpu = gpu_range[idx] 
    cmd += 'python wrapper.py ' \
          '-start={} -end={} -out_file=./job_logs/job_{:03d}.txt -gpu={} '.format(start, end, idx, gpu)
    
    if idx != (n_jobs-1):
        cmd+= ' & '

#import pdb; pdb.set_trace()
os.system(cmd)
