import os
import time
import argparse
import json
import numpy as np
import shutil

from collections import Counter

# paths and arguments
blend_files_path = '../blend_files'

param_json_path = 'data_generation_parameters.json'

with open(param_json_path, 'r') as load_file:
    data_gen_params = json.load(load_file)        

blender_path = data_gen_params['paths']['blender_path']

blender_script_path = os.path.abspath('generate.py')
blendfile_path = os.path.abspath(os.path.join(blend_files_path, 'empty_scene.blend'))

parser = argparse.ArgumentParser(description='Range of Objects')
parser.add_argument('-start', type=int, help='start point in data list', default=0)
parser.add_argument('-end', type=int, help='end point in data list', default=200)
parser.add_argument('-out_file', type=str, help='file to output progress to', required=True)
parser.add_argument('-v', dest='v', action='store_true', help='verbose: print or supress blender output', default=False)

parser.add_argument('-gpu', type=int, help='gpu index to use', required=True)

args, unknown = parser.parse_known_args()
out_file = args.out_file

target_directory = data_gen_params['paths']['ABC_path']

obj_paths = []
for root, dirs, files in os.walk(target_directory):
    for name in files:
        fpath = os.path.join(root, name)

        data_type = fpath.split('.')[-1]
        if data_type == 'obj':
            obj_paths.append(fpath)

obj_paths = np.array(obj_paths)
global_time = time.time()

### Creating output directory + Moving the data_generation_params config file 
output_dir = data_gen_params['paths']['output_path']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)   

shutil.copy(param_json_path, 
            os.path.join(output_dir, 
                         os.path.split(param_json_path)[1]
                ) 
            )

for i in range(args.start, args.end):
    p = obj_paths[i]
    obj_num = p.split('/')[-1]
    
    start_time = time.time()
    
    if args.v:
        cmd = "{} -noaudio --background {} --python {} -- " \
            "--obj_fname {} --gpu {}".format(
                    blender_path,
                    blendfile_path,
                    blender_script_path,
                    obj_num,
                    args.gpu)
    else:
        cmd = "{} -noaudio --background {} --python {} -- " \
            "--obj_fname {} --gpu {} 1>/tmp/out.txt".format(
                    blender_path,
                    blendfile_path,
                    blender_script_path,
                    obj_num,
                    args.gpu)

    os.system(cmd)
    with open(out_file, 'a') as f:
        out_str = "--- {:.2f} seconds for obj {} [{}/{}]---".format(time.time() - start_time, 
                                                                obj_num, 
                                                                i, 
                                                                args.end)
        out_str += '\n' 
        if i % 50 == 0:
            out_str += "Time since start is {:.2f} minutes".format(
                                                    (time.time() - global_time)/60)
            out_str += '\n'

        f.write(out_str)

total_time = time.time() - global_time

with open(out_file, 'a') as f:
    out_str = "Total time {:.2f}".format(total_time/60) 
    out_str += '\n'
    f.write(out_str)

