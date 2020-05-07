import numpy as np
import os
import config
from datetime import datetime
import copy
import torch
from mesh_gen_utils.libmise import MISE
from mesh_gen_utils.libmesh import check_mesh_contains
from mesh_gen_utils import libmcubes
import trimesh
from mesh_gen_utils.libkdtree import KDTree
from torch.autograd import Variable
import h5py
import torch.nn as nn
import struct
import pymesh

def writelogfile(log_dir):
    log_file_name = os.path.join(log_dir, 'log.txt')
    with open(log_file_name, "a+") as log_file:
        log_string = get_log_string()
        log_file.write(log_string)


def get_log_string():
    now = str(datetime.now().strftime("%H:%M %d-%m-%Y"))
    log_string = ""
    log_string += " -------- Hyperparameters and settings -------- \n"
    log_string += "{:25} {}\n".format('Time:', now)
    log_string += "{:25} {}\n".format('Mini-batch size:', \
        config.training['batch_size'])
    log_string += "{:25} {}\n".format('Batch size eval:', \
        config.training['batch_size_eval'])
    log_string += "{:25} {}\n".format('Num epochs:', \
        config.training['num_epochs'])
    log_string += "{:25} {}\n".format('Out directory:', \
        config.training['out_dir'])
    log_string += "{:25} {}\n".format('Random view:', \
        config.data_setting['random_view'])
    log_string += "{:25} {}\n".format('Sequence length:', \
        config.data_setting['seq_len'])
    log_string += "{:25} {}\n".format('Input size:', \
        config.data_setting['input_size'])
    log_string += " -------- Data paths -------- \n"
    log_string += "{:25} {}\n".format('Dataset path', \
        config.path['src_dataset_path'])
    log_string += "{:25} {}\n".format('Point path', \
        config.path['src_pt_path'])
    log_string += " ------------------------------------------------------ \n"
    return log_string




def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1_temp = copy.deepcopy(occ1)
    occ2_temp = copy.deepcopy(occ2)
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    # Avoid dividing by 0
    if (area_union == 0).any():
        return 0.

    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou

def compute_acc(sdf_pred, sdf, thres=0.01, iso=0.003):
    '''
    This function computes metric for sdf representation
    Args:
        sdf_pred: predicted sdf values
        sdf: gt sdf values
        thres: threshold to compute accuracy
        iso: iso value when generating ground truth meshes
    Returns:
        acc_sign: sign IoU where the signs of sdf_pred matches with sdf
        acc_thres: portion of points where sdf_pred is within thres from sdf
        iou: regular point IoU
    '''
    sdf_pred = np.asarray(sdf_pred)
    sdf = np.asarray(sdf)

    acc_sign = (((sdf_pred-iso) * (sdf-iso)) > 0).mean(axis=-1)
    acc_sign = np.mean(acc_sign, axis=0)

    occ_pred = sdf_pred <= iso
    occ = sdf <= iso

    iou = compute_iou(occ_pred, occ)

    acc_thres = (np.abs(sdf_pred-sdf) <= thres).mean(axis=-1)
    acc_thres = np.mean(acc_thres, axis=0)
    return acc_sign, acc_thres, iou[0]

def get_sdf_h5(sdf_h5_file):
    '''
    This function reads sdf files saved in h5 format
    '''
    h5_f = h5py.File(sdf_h5_file, 'r')
    try:
        if ('pc_sdf_original' in h5_f.keys()
                and 'pc_sdf_sample' in h5_f.keys()
                and 'norm_params' in h5_f.keys()):
            ori_sdf = h5_f['pc_sdf_original'][:].astype(np.float32)
            sample_sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
            ori_pt = ori_sdf[:,:3]
            ori_sdf_val = None
            if sample_sdf.shape[1] == 4:
                sample_pt, sample_sdf_val = sample_sdf[:,:3], sample_sdf[:,3]
            else:
                sample_pt, sample_sdf_val = None, sample_sdf[:, 0]
            norm_params = h5_f['norm_params'][:]
            sdf_params = h5_f['sdf_params'][:]
        else:
            raise Exception('no sdf and sample')
    finally:
        h5_f.close()
    return ori_pt, ori_sdf_val, sample_pt, sample_sdf_val, norm_params, sdf_params

def apply_rotate(input_points, rotate_dict):
    '''
    Azimuth rotation and elevation
    '''
    theta_azim = rotate_dict['azim']
    theta_elev = rotate_dict['elev']
    theta_azim = np.pi+theta_azim/180*np.pi
    theta_elev = theta_elev/180*np.pi
    r_elev = np.array([[1,       0,          0],
                        [0, np.cos(theta_elev), -np.sin(theta_elev)],
                        [0, np.sin(theta_elev), np.cos(theta_elev)]])
    r_azim = np.array([[np.cos(theta_azim), 0, np.sin(theta_azim)],
                        [0,               1,       0],
                        [-np.sin(theta_azim),0, np.cos(theta_azim)]])

    rotated_points = r_elev@r_azim@input_points.T
    return rotated_points.T

def sample_points(input_points, input_vals, num_points):
    '''
    Samples a subset of points
    Args:
        input_points: 3D coordinates of points
        input_vals: corresponding occ/sdf values
        num_points: number of points
    Returns:
        selected_points: 3D coordinates of subset of points 
        selected_vals: corresponding occ/sdf values of selected points
    '''
    if num_points != -1:
        idx = torch.randint(len(input_points), size=(num_points,))
    else:
        idx = torch.arange(len(input_points))
    selected_points = input_points[idx, :]
    selected_vals = input_vals[idx]
    return selected_points, selected_vals

def LpLoss(logits, sdf, p=1, thres=0.01, weight=4.):
    '''
    Customed Lp loss for SDFNet
    Args:
        logits: logits from model
        sdf: ground truth sdf
        p: degree of loss, default is 1 for L1 loss
        thres: threshold to apply weights
        weight: weight applied on points within thres distance to the surface
    '''
    sdf = Variable(sdf.data, requires_grad=False).cuda()
    loss = torch.abs(logits-sdf).pow(p).cuda()
    weight_mask = torch.ones(loss.shape).cuda()
    weight_mask[torch.abs(sdf) < thres] =\
             weight_mask[torch.abs(sdf) < thres]*weight 
    loss = loss * weight_mask
    loss = torch.sum(loss, dim=-1, keepdim=False)
    loss = torch.mean(loss)
    return loss

def generate_mesh(img, points, model, threshold=0.2, box_size=1.7, \
            resolution0=16, upsampling_steps=2):
    '''
    Generates mesh for occupancy representations using MISE algorithm
    '''
    model.eval()

    threshold = np.log(threshold) - np.log(1. - threshold)
    mesh_extractor = MISE(
        resolution0, upsampling_steps, threshold)
    p = mesh_extractor.query()

    with torch.no_grad():
        feats = model.encoder(img)

    while p.shape[0] != 0:
        pq = torch.FloatTensor(p).cuda()
        pq = pq / mesh_extractor.resolution

        pq = box_size * (pq - 0.5)

        with torch.no_grad():
            pq = pq.unsqueeze(0)
            occ_pred = model.decoder(pq, feats)
        values = occ_pred.squeeze(0).detach().cpu().numpy()
        values = values.astype(np.float64)
        mesh_extractor.update(p, values)

        p = mesh_extractor.query()
    value_grid = mesh_extractor.to_dense()

    mesh = extract_mesh(value_grid, feats, box_size, threshold)
    return mesh

def extract_mesh(value_grid, feats, box_size, threshold):
    '''
    Extract mesh helper function for generating mesh for occupancy \
        representations
    '''
    n_x, n_y, n_z = value_grid.shape
    value_grid_padded = np.pad(
            value_grid, 1, 'constant', constant_values=-1e6)
    vertices, triangles = libmcubes.marching_cubes(
            value_grid_padded, threshold)
    # Shift back vertices by 0.5
    vertices -= 0.5
    # Undo padding
    vertices -= 1
    # Normalize
    vertices /= np.array([n_x-1, n_y-1, n_z-1])
    vertices = box_size * (vertices - 0.5)

    # Create mesh
    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    return mesh

def eval_mesh(mesh, pointcloud_gt, normals_gt, points, val_gt, \
                n_points=300000, rep='occ', \
                sdf_val=None, iso=0.003):
    '''
    Computes metric on generated mesh
    Args:
        mesh: generated mesh
        pointcloud_gt: ground truth pointcloud (dimension Nx3)
        normals_gt: ground truth normals (dimension Nx3)
        points: 3D coordinates of points (dimension Nx3)
        val_gt: ground truth occ/sdf
        n_points: number of points to sample from generated mesh
        rep: representation, can be either occ or sdf
        sdf_val: predicted sdf values
        iso: isosurface value used in ground truth mesh generation
    Returns:
        metric dictionary contains:
            iou: regular point IoU for occ; regular point IoU with sign IoU \
                for sdf
            cd: Chamfer distance
            completeness: mesh completeness d(target->pred)
            accuracy: mesh accuracy d(pred->target)
            normals_completeness: normals completeness measurement
            normals_accuracy: normals accuracy measurement
            normals: Normal Consistency
            fscore: Fscore from 6 different thresholds
            precision: accuracy < threshold
            recall: completeness < threshold
    '''

    if mesh is not None and type(mesh)==trimesh.base.Trimesh and len(mesh.vertices) != 0 and len(mesh.faces) != 0:
        pointcloud, idx = mesh.sample(n_points, return_index=True)
        pointcloud = pointcloud.astype(np.float32)
        normals = mesh.face_normals[idx]
    else:
        return {'iou': 0., 'cd': 2*np.sqrt(3), 'completeness': np.sqrt(3),\
                    'accuracy': np.sqrt(3), 'normals_completeness': -1,\
                    'normals_accuracy': -1, 'normals': -1, \
                    'fscore': np.zeros(6, dtype=np.float32), \
                    'precision': np.zeros(6, dtype=np.float32), \
                    'recall': np.zeros(6, dtype=np.float32)}
    # Eval pointcloud
    pointcloud = np.asarray(pointcloud)
    pointcloud_gt = np.asarray(pointcloud_gt.squeeze(0))
    normals = np.asarray(normals)
    normals_gt = np.asarray(normals_gt.squeeze(0))

    # Completeness: how far are the points of the target point cloud
    # from the predicted point cloud
    completeness, normals_completeness = distance_p2p(
            pointcloud_gt, normals_gt, pointcloud, normals)

    # Accuracy: how far are the points of the predicted pointcloud
    # from the target pointcloud
    accuracy, normals_accuracy = distance_p2p(
        pointcloud, normals, pointcloud_gt, normals_gt
    )

    # Get fscore
    fscore_array, precision_array, recall_array = [], [], []
    for i, thres in enumerate([0.5, 1, 2, 5, 10, 20]):
        fscore, precision, recall = calculate_fscore(\
            accuracy, completeness, thres/100.)
        fscore_array.append(fscore)
        precision_array.append(precision)
        recall_array.append(recall)
    fscore_array = np.array(fscore_array, dtype=np.float32)
    precision_array = np.array(precision_array, dtype=np.float32)
    recall_array = np.array(recall_array, dtype=np.float32)

    accuracy = accuracy.mean()
    normals_accuracy = normals_accuracy.mean()

    completeness = completeness.mean()
    normals_completeness = normals_completeness.mean()

    cd = completeness + accuracy
    normals = 0.5*(normals_completeness+normals_accuracy)

    # Compute IoU
    if rep == 'occ':
        occ_mesh = check_mesh_contains(mesh, points.cpu().numpy().squeeze(0))
        iou = compute_iou(occ_mesh, val_gt.cpu().numpy().squeeze(0))
    else:

        occ_mesh = check_mesh_contains(mesh, points.cpu().numpy().squeeze(0))
        val_gt_np = val_gt.cpu().numpy()
        occ_gt = val_gt_np <= iso
        iou = compute_iou(occ_mesh, occ_gt) 

        # sdf iou
        sdf_iou, _, _ = compute_acc(sdf_val.cpu().numpy(),\
                        val_gt.cpu().numpy()) 
        iou = np.array([iou[0], sdf_iou])

    return {'iou': iou, 'cd': cd, 'completeness': completeness,\
                'accuracy': accuracy, \
                'normals_completeness': normals_completeness,\
                'normals_accuracy': normals_accuracy, 'normals': normals, \
                'fscore': fscore_array, 'precision': precision_array,\
                'recall': recall_array}

def calculate_fscore(accuracy, completeness, threshold):
    '''
    Calculate FScore given accuracy, completeness and threshold
    '''
    recall = np.sum(completeness < threshold)/len(completeness)
    precision = np.sum(accuracy < threshold)/len(accuracy)
    if precision + recall > 0:
        fscore = 2*recall*precision/(recall+precision)
    else:
        fscore = 0
    return fscore, precision, recall


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src: source points
        normals_src: source normals
        points_tgt: target points
        normals_tgt: target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def generate_mesh_sdf(img, model, obj_path, sdf_path, iso=0.003, box_size=1.01, resolution=64):
    '''
    Generates mesh for sdf representation
    '''
    # create cube
    min_box = -box_size/2
    max_box = box_size/2
    x_ = np.linspace(min_box, max_box, resolution+1)
    y_ = np.linspace(min_box, max_box, resolution+1)
    z_ = np.linspace(min_box, max_box, resolution+1)

    z, y, x = np.meshgrid(z_, y_, x_, indexing='ij')
    x = np.expand_dims(x, 3)
    y = np.expand_dims(y, 3)
    z = np.expand_dims(z, 3)
    all_pts = np.concatenate((x, y, z), axis=3).astype(np.float32)
    all_pts = all_pts.reshape(1, -1, 3)

    all_pts = Variable(torch.FloatTensor(all_pts)).cuda()

    pred_sdf = model(all_pts, img)
    pred_sdf = pred_sdf.data.cpu().numpy().reshape(-1)

    f_sdf_bin = open(sdf_path, 'wb')
    f_sdf_bin.write(struct.pack('i', -(resolution)))  # write an int
    f_sdf_bin.write(struct.pack('i', (resolution)))  # write an int
    f_sdf_bin.write(struct.pack('i', (resolution)))  # write an int

    pos = np.array([min_box, min_box, min_box, max_box, max_box, max_box])

    positions = struct.pack('d' * len(pos), *pos)
    f_sdf_bin.write(positions)
    val = struct.pack('=%sf'%pred_sdf.shape[0], *(pred_sdf))
    f_sdf_bin.write(val)
    f_sdf_bin.close()

    marching_cube_cmd = "./isosurface/computeMarchingCubes" + " " + \
                 sdf_path + " " + obj_path + " -i " + str(iso)
    os.system(marching_cube_cmd)










