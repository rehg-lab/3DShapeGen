#import config
import numpy as np
import os
import pymesh
import trimesh
import json
from multiprocessing import Pool
from skimage import measure


mesh_dir = 'ShapeNet_sdf_meshes_13'
# mesh_dir = 'ShapeNet_sdf_meshes_42'

data_split_json_path = 'val_LRBg_study.json'
save_dir = 'ShapeNet_LRBg_val_ptcl'
pointcloud_size = 100000
num_split = 1

def generate_ptcld(arg):
    split, categories = arg

    for cat in categories:
        #if cat == '04401088' or cat == '04530566' or cat == '04379243' or cat == '04256520' or cat == '04090263':
        #    continue
        objects = split[cat]
        cat_path = os.path.join(mesh_dir, cat)
        cat_save_path = os.path.join(save_dir, cat)
        os.makedirs(cat_save_path, exist_ok=True)
        for obj in objects:
            #if cat == '03691459' and obj == '6c71c0791b014bbe7ac477bac77def9':
            #    continue
            print('Cat: ', cat, ' Object: ', obj)
            os.makedirs(os.path.join(cat_save_path,obj), exist_ok=True)

            ############ Voxel (only for genre)
            '''
            for i in range(20):

                vox_path = os.path.join(cat_path, obj, "vox_NPZ/{:04d}.npz".format(i))
                try:
                    vox = np.load(vox_path)['voxel']
                except Exception:
                    print("ERROR")
                    break
                verts, faces, normals, values = measure.marching_cubes_lewiner(
                    vox, 0.5, spacing=(1 / 128, 1 / 128, 1 / 128))
                ob = trimesh.Trimesh(vertices=verts - 0.5, faces=faces)


                save_file_name = os.path.join(cat_save_path, obj, '{:04d}_pointcloud.npz'.format(i))
                if os.path.exists(save_file_name):
                    print('Existed')
                    continue
                obj_path = os.path.join(cat_path, obj, '{:04d}_isosurf.obj'.format(i))
                ob = trimesh.load(obj_path)


                ptcl, face_idx = ob.sample(pointcloud_size, return_index=True)
                normals = ob.face_normals[face_idx]

                ptcl = ptcl.astype(np.float16)
                normals = normals.astype(np.float16)

                np.savez_compressed(save_file_name, points=ptcl, normals=normals)
            '''


            #####################################################
            try:
                obj_path = os.path.join(cat_path, obj, 'isosurf.obj')

                save_file_name = os.path.join(cat_save_path, obj, 'pointcloud.npz')

                if os.path.exists(save_file_name):
                    print('Existed')
                    continue
                ob = trimesh.load(obj_path)
                ptcl, face_idx = ob.sample(pointcloud_size, return_index=True)
                normals = ob.face_normals[face_idx]

                ptcl = ptcl.astype(np.float16)
                normals = normals.astype(np.float16)

                np.savez_compressed(save_file_name, points=ptcl, normals=normals)
            except Exception:
                print('Dies')
            #####################################################


with open(data_split_json_path, 'r') as data_split_file:
    data_splits = json.load(data_split_file)
mode_array = ['test']
for mode in mode_array:
    print('Doing mode %s'%(mode))
    split = data_splits[mode]

    categories = sorted(list(split.keys()))
    # categories = sorted([c for c in categories if c in config.data_setting['categories']])
    split_ = np.asarray([split for _ in range(num_split)])
    categories_split = np.array_split(categories, num_split)
    pool = Pool(num_split)
    pool.map(generate_ptcld, zip(split_, categories_split))






