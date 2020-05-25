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
obj_json_path = 'obj_dict.json'

with open(param_json_path, 'r') as load_file:
    data_gen_params = json.load(load_file)        

with open(obj_json_path, 'r') as json_file:
    obj_dict = json.load(json_file)

blender_path = data_gen_params['paths']['blender_path']

blender_script_path = os.path.abspath('generate.py')
blendfile_path = os.path.abspath(os.path.join(blend_files_path, 'empty_scene.blend'))

parser = argparse.ArgumentParser(description='Range of Objects')
parser.add_argument('-start', type=int, help='start point in data list', default=0)
parser.add_argument('-end', type=int, help='end point in data list', default=200)
parser.add_argument('-out_file', type=str, help='file to output progress to', required=True)
parser.add_argument('-gpu', type=int, help='gpu index to use', required=True)
parser.add_argument('-v', dest='v', action='store_true', default=False)

args, unknown = parser.parse_known_args()
out_file = args.out_file

global_time = time.time()
i = 0

synset_list_13 = ['02691156', '02828884', '02933112',
                  '02958343', '03001627', '03211117',
                  '03636649', '03691459', '04090263',
                  '04256520', '04379243', '04401088',
                  '04530566']

synset_list_42 = ['02747177', '02801938', '02818832', '02871439', '02880940',
                  '02942699', '02954340', '03046257', '03207941', '03325088',
                  '03467517', '03593526', '03642806', '03759954', '03790512',
                  '03928116', '03948459', '04004475', '04099429', '04330267',
                  '04468005', '02773838', '02808440', '02843684', '02876657',
                  '02924116', '02946921', '02992529', '03085013', '03261776',
                  '03337140', '03513137', '03624134', '03710193', '03761084',
                  '03797390', '03938244', '03991062', '04074963', '04225987',
                  '04460130', '04554684']


new_dict = {}


obj_dict = {key:obj_dict[key] for key  in sorted(obj_dict.keys()) if obj_dict[key][0] in synset_list_13}
obj_dict = {i:obj_dict[key] for i, key  in enumerate(sorted(obj_dict.keys())) }

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
    synset, obj = obj_dict[i]
    start_time = time.time()
    
    #objects that stall the generating process
    if obj in ['3975b2350688e38c65552c4ac8607d25',
               'c5c4e6110fbbf5d3d83578ca09f86027']:
        continue
    
    if args.v:
         cmd = "{} -noaudio --background {} --python {} -- " \
                "--obj_n {} " \
                "--synset {} " \
                "--gpu {}".format(
                        blender_path, 
                        blendfile_path, 
                        blender_script_path, 
                        obj, 
                        synset,
                        args.gpu)
       
    else:
        cmd = "{} -noaudio --background {} --python {} -- "  \
                "--obj_n {} " \
                "--synset {} " \
                "--gpu {} 1>/tmp/out.txt".format(
                        blender_path, 
                        blendfile_path, 
                        blender_script_path, 
                        obj, 
                        synset,
                        args.gpu)


    os.system(cmd)
    with open(out_file, 'a') as f:
        out_str = "--- {:.2f} seconds for obj {} [{}/{}]---".format(time.time() - start_time, 
                                                                obj, 
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

