import numpy as np
import os
import pymesh
import trimesh
import json
from multiprocessing import Pool
from skimage import measure
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--mesh_dir', type=str, default='.', \
    help='Orginal mesh directory')
parser.add_argument('--json_path', type=str, default='.', \
    help='Path to json file')
parser.add_argument('--save_dir', type=str, default='.', \
    help='Where to save pointcloud')
parser.add_argument('--pointcloud_size', type=int, default=100000, \
    help='Where to save pointcloud')
parser.add_argument('--num_split', type=int, default=12, \
    help='Number of threads to use')
parser.add_argument('--mode', type=str, default=None, \
    help='Generating mode (train, val, test). If None all 3 are generated')
args = parser.parse_args()


def generate_ptcld(arg):
    split, categories = arg

    for cat in categories:
        objects = split[cat]
        cat_path = os.path.join(mesh_dir, cat)
        cat_save_path = os.path.join(save_dir, cat)
        os.makedirs(cat_save_path, exist_ok=True)
        for obj in objects:
            os.makedirs(os.path.join(cat_save_path,obj), exist_ok=True)
            try:
                obj_path = os.path.join(cat_path, obj, 'isosurf.obj')

                save_file_name = os.path.join(cat_save_path, obj, 'pointcloud.npz')

                if os.path.exists(save_file_name):
                    continue

                # Load gt mesh and sample pointclouds
                ob = trimesh.load(obj_path)
                ptcl, face_idx = ob.sample(pointcloud_size, return_index=True)
                normals = ob.face_normals[face_idx]

                ptcl = ptcl.astype(np.float16)
                normals = normals.astype(np.float16)

                np.savez_compressed(save_file_name, points=ptcl, \
                    normals=normals)
            except Exception:
                print('Error in generating pointcloud for category %s, \
                    object %s'%(cat, obj))



if __name__ == '__main__':
    mesh_dir = args.mesh_dir
    json_path = args.json_path
    save_dir = args.save_dir
    pointcloud_size = args.pointcloud_size
    num_split = args.num_split
    mode_array = args.mode
    if mode_array == None:
        mode_array = ['train', 'val', 'test']
    else:
        mode_array = [mode_array]

    with open(json_path, 'r') as data_split_file:
        data_splits = json.load(data_split_file)
    for mode in mode_array:
        print('Doing mode %s'%(mode))
        split = data_splits[mode]
        categories = sorted(list(split.keys()))
        split_ = np.asarray([split for _ in range(num_split)])
        categories_split = np.array_split(categories, num_split)
        pool = Pool(num_split)
        pool.map(generate_ptcld, zip(split_, categories_split))






