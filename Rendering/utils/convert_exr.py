import numpy as np
import OpenEXR as exr
import Imath
import matplotlib.pyplot as plt
import os
import argparse
from joblib import Parallel, delayed 

parser = argparse.ArgumentParser()                                                         
parser.add_argument('--data_path', type=str, help='path to dataset to overlay backgrounds')
args = parser.parse_args()                                                                 
                                                                                           
src_dataset_path = args.data_path                                                          

def make_dir(directory):
    if not os.path.exists(directory):
            os.makedirs(directory)
                                                                                           
exr_file_paths = []                                                                        
for dirname, subdirs, files in os.walk(src_dataset_path):                                  
    for fname in files:                                                                    
        fpath = os.path.join(dirname, fname)                                               
        if 'openEXR_output' in fpath:
            obj_id = fpath.split('/')[-3]
            synset_id = fpath.split('/')[-4]
            exr_file_paths.append([fpath, obj_id, synset_id])                                                   

def readEXR(filename):
    """Read RGB + Depth data from EXR image file.
    Parameters
    ----------
    filename : str
        File path.
    Returns
    -------
    img : RGB image in float32 format.
    Z : Depth buffer in float3.
    """
    
    exrfile = exr.InputFile(filename)
    dw = exrfile.header()['dataWindow']
    isize = (dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1)
    
    channels = ['R', 'G', 'B']
    channelData = dict()
    
    for c in channels:
        C = exrfile.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
        C = np.fromstring(C, dtype=np.float32)
        C = np.reshape(C, isize)
        
        channelData[c] = C
    
    
    img = np.concatenate([channelData[c][...,np.newaxis] for c in ['R', 'G', 'B']], axis=2)
     
    return img

def job(arg):
    filename, obj_id, syn_id = arg
    
    target_dir = os.path.join(args.data_path, syn_id, obj_id, 'depth_NPZ')

    num = filename.split('/')[-1].split('.')[0]
    target_path = os.path.join(target_dir, num+'.npz')

    if os.path.exists(target_path):
        return 0 
    
    try:
        make_dir(target_dir)

        img = readEXR(filename)

        img_vals = img[np.where(img!=np.max(img))].flatten()

        max_depth = np.max(img_vals)
        min_depth = np.min(img_vals)

        img[img == 1e10] = max_depth
        img_0_1 = (1-(img-min_depth)/(max_depth-min_depth))
        
        np.savez_compressed(target_path, min_max = [min_depth, max_depth], img = img_0_1[:,:,0])
        return 1
    except:
        print(arg, 'failure')
        return (obj_id, syn_id)

results = Parallel(n_jobs=12, verbose=1, backend="multiprocessing")(map(delayed(job), exr_file_paths))

np.save(os.path.join(src_dataset_path, 'problematic_1.npy'), 
        np.array([x for x in results if x != 0]))
