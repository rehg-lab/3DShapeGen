import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import glob
import config
import utils
import json
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, num_points=-1,mode='train',rep='occnet',\
            coord_system='2dvc'):

        self.mode = mode
        self.coord_system = coord_system
        
        self.input_size = config.data_setting['input_size']
        self.num_points = num_points

        self.src_dataset_path = config.path['src_dataset_path']
        self.input_image_path = config.path['input_image_path']
        self.input_depth_path = config.path['input_depth_path']
        self.input_normal_path = config.path['input_normal_path']
        self.input_seg_path = config.path['input_seg_path']

        self.img_extension = config.data_setting['img_extension']


        self.src_pt_path = config.path['src_pt_path']
        self.input_points_path = config.path['input_points_path']
        self.input_pointcloud_path = config.path['input_pointcloud_path']

        self.input_metadata_path = config.path['input_metadata_path']


        self.data_split_json_path = config.path['data_split_json_path']

        self.rep = rep

        self.categories = config.data_setting['categories']

        with open(self.data_split_json_path, 'r') as data_split_file:
            self.data_splits = json.load(data_split_file)
        self.split = self.data_splits[self.mode]
        self.random_view = config.data_setting['random_view']
        self.seq_len = config.data_setting['seq_len']

        self.catnames = sorted(list(self.split.keys()))

        # When categories is specified
        if self.categories is not None:
            self.catnames = sorted([c for c in self.catnames \
                            if c in self.categories])

        self.obj_cat_map = [(obj,cat) for cat in self.catnames \
                                for obj in self.split[cat] \
                            if os.path.exists(os.path.join( \
                                self.src_dataset_path, cat,obj))]

        # Get all image paths
        self.img_paths = [os.path.join(self.src_dataset_path, cat, obj) \
                                    for (obj, cat) in self.obj_cat_map \
                                if os.path.exists(\
                                    os.path.join(self.src_dataset_path, \
                                        cat, obj))]
        # Get all sdf paths
        self.sdf_h5_paths = [os.path.join(self.src_pt_path, cat, obj, \
                        'ori_sample.h5') for (obj, cat) in self.obj_cat_map \
                        if os.path.exists(os.path.join( \
                            self.src_dataset_path, cat, obj))]
        # Get all pointcloud paths
        self.pointcld_split_paths = [os.path.join(self.src_pt_path, cat, obj) \
                            for (obj, cat) in self.obj_cat_map if \
                            os.path.exists(os.path.join(\
                                self.src_dataset_path, cat, obj))]
        # Get all metadata paths               
        self.metadata_split_paths = [os.path.join(self.src_dataset_path, cat, \
                        obj, 'metadata.txt') \
                        for (obj, cat) in self.obj_cat_map \
                        if os.path.exists(os.path.join(\
                            self.src_dataset_path, cat, obj))]

        if self.coord_system == '3dvc':
            # Get initial pose for 3dvc
            self.hvc_metadata_split_paths = [\
                os.path.join(self.src_dataset_path, cat, \
                    obj, 'hard_vc_metadata.txt') \
                            for (obj, cat) in self.obj_cat_map \
                            if os.path.exists(os.path.join(self.src_dataset_path, cat, obj))]

        self.img_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(\
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_split_paths, self.depth_split_paths, \
            self.normal_split_paths, self.seg_split_paths = \
                    self.load_image_paths()

    def load_image_paths(self):
        '''
        Loads all images paths

        '''
        if self.input_image_path is not None:
            image_split_files = [sorted(glob.glob(
                os.path.join(\
                obj_path, self.input_image_path, '*.%s' \
                    % self.img_extension)))[:] \
                        for obj_path in self.img_paths \
                            if os.path.exists(\
                                os.path.join(obj_path, self.input_image_path))]
        else:
            image_split_files = None

        if self.input_depth_path is not None:
            seg_split_files = [sorted(glob.glob(
                os.path.join(obj_path, self.input_seg_path, '*.%s' \
                    % self.img_extension)))[:] \
                        for obj_path in self.img_paths \
                            if os.path.exists(\
                                os.path.join(obj_path, self.input_seg_path))]

            depth_split_files = [sorted(glob.glob(
                    os.path.join(obj_path, self.input_depth_path, \
                        '*.npz')))[:] for obj_path in self.img_paths \
                            if os.path.exists(\
                                os.path.join(obj_path, self.input_depth_path))]
        else:
            depth_split_files = None
            seg_split_files = None
        if self.input_normal_path is not None:
            normal_split_files = [sorted(glob.glob(
                os.path.join(obj_path, self.input_normal_path, '*.%s' \
                    % self.img_extension)))[:] \
                        for obj_path in self.img_paths \
                            if os.path.exists(\
                            os.path.join(obj_path, self.input_normal_path))]
        else:
            normal_split_files = None

        return image_split_files, depth_split_files, \
            normal_split_files, seg_split_files

    def get_data_sample(self, index, img_idx=-1):
        if self.random_view:
            assert img_idx != -1
        else:
            idx = index//self.seq_len
            img_idx = index % self.seq_len

            index = idx

        if self.image_split_paths is not None:

            input_image = self.image_split_paths[index][img_idx]
            image_data = Image.open(input_image).convert('RGB')
            image_data = self.img_transform(image_data)
            image_data = np.array(image_data.numpy())
        else:
            image_data = np.array([])

        
        
        if self.depth_split_paths is not None:
            input_depth = self.depth_split_paths[index][img_idx]
            input_seg = self.seg_split_paths[index][img_idx]
            depth_data = np.load(input_depth)['img']
            depth_min_max = np.load(input_depth)['min_max']
            min_d, max_d = depth_min_max[0], depth_min_max[1]

            ########################################
            depth_image = Image.fromarray(np.uint8(depth_data*255.))
            depth_image = depth_image.resize(\
                (self.input_size, self.input_size))            
            depth_data = np.array(depth_image)/255.
            ########################################
            # Convert depth to range min to max
            depth_data = 1 - depth_data
            seg_data = Image.open(input_seg).convert('L')
            seg_data = seg_data.resize((self.input_size, self.input_size))
            seg_data = np.array(seg_data)/255. # 0-1 with object as 1

            depth_data[seg_data == 0.] = 10. # Set background to max value

            depth_data[seg_data != 0.] = depth_data[seg_data != 0.]*(max_d-min_d)+min_d
            depth_data = np.expand_dims(depth_data, axis=2)
            depth_data = depth_data.transpose(2,0,1)

            if len(image_data) == 0:
                image_data = depth_data
            else:
                image_data = np.concatenate(\
                    [image_data, depth_data],axis=0)

        else:
            depth_data = None
        if self.normal_split_paths is not None:
            input_normal = self.normal_split_paths[index][img_idx]

            normal_data = Image.open(input_normal).convert('RGB')
            normal_data = normal_data.resize(\
                (self.input_size, self.input_size))
            normal_data = np.array(normal_data)/255.
            normal_data = normal_data.transpose(2,0,1)
            if len(image_data) == 0:
                image_data = normal_data
            else: 
                image_data = np.concatenate(\
                    [image_data, normal_data],axis=0)
        else:
            normal_data = None

        image_data = torch.FloatTensor(image_data)

        return image_data


    def get_points_sample(self, index, img_idx=-1):
        if self.random_view:
            assert img_idx != -1
        else:
            idx = index//self.seq_len
            img_idx = index % self.seq_len

            index = idx
        input_points_path = self.points_split_paths[index]
        input_points = np.load(\
                        os.path.join(input_points_path,'points.npz'),\
                            mmap_mode='r')['points']
        input_occs = np.load(\
                        os.path.join(input_points_path, 'occupancies.npz'),\
                            mmap_mode='r')['occupancies']
        input_occs = np.unpackbits(input_occs)
        input_occs = input_occs.astype(np.float32)

        input_points, input_occs = utils.sample_points(input_points,\
                                                   input_occs,\
                                                   self.num_points)

        if self.coord_system == '2dvc':
            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]}

            input_points = utils.apply_rotate(input_points, rotate_dict)
        elif self.coord_system == '3dvc':

            input_hvc_meta_path = self.hvc_metadata_split_paths[index]
            hvc_meta = np.loadtxt(input_hvc_meta_path)
            hvc_rotate_dict = {'elev': hvc_meta[1], 'azim': hvc_meta[0]}
            input_points = utils.apply_rotate(input_points, hvc_rotate_dict)

            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]-180}

            input_points = utils.apply_rotate(input_points, rotate_dict)


        input_points = torch.FloatTensor(input_points)
        input_occs = torch.FloatTensor(input_occs)
        return input_points, input_occs

    def get_points_sdf_sample(self, index, img_idx=-1):
        assert self.sdf_h5_paths is not None
        if self.random_view:
            assert img_idx != -1
        else:
            idx = index//self.seq_len
            img_idx = index % self.seq_len

            index = idx
        ori_pt, ori_sdf_val, input_points, input_sdfs, norm_params, \
            sdf_params  = utils.get_sdf_h5(self.sdf_h5_paths[index])
        input_points, input_sdfs = utils.sample_points(input_points,\
                                           input_sdfs,\
                                           self.num_points)

        if self.coord_system == '2dvc':

            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]}

            input_points = utils.apply_rotate(input_points, rotate_dict)

        elif self.coord_system == '3dvc':

            input_hvc_meta_path = self.hvc_metadata_split_paths[index]
            hvc_meta = np.loadtxt(input_hvc_meta_path)
            hvc_rotate_dict = {'elev': hvc_meta[1], 'azim': hvc_meta[0]}
            input_points = utils.apply_rotate(input_points, hvc_rotate_dict)

            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]-180}

            input_points = utils.apply_rotate(input_points, rotate_dict)

        input_points = torch.FloatTensor(input_points)
        input_sdfs = torch.FloatTensor(input_sdfs)
        return input_points, input_sdfs


    def get_pointcloud_sample(self, index, img_idx=-1):
        if self.random_view:
            assert img_idx != -1
        else:
            idx = index//self.seq_len
            img_idx = index % self.seq_len

            index = idx
        input_pointcld_path = self.pointcld_split_paths[index]
        input_pointcld_path = os.path.join(input_pointcld_path,\
                                    'pointcloud.npz')
        input_ptcld_dict = np.load(input_pointcld_path, mmap_mode='r')
        input_pointcld = input_ptcld_dict['points'].astype(np.float32)
        input_normals = input_ptcld_dict['normals'].astype(np.float32)

        if self.coord_system == '2dvc':

            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]}

            input_pointcld = utils.apply_rotate(input_pointcld, rotate_dict)
            input_normals = utils.apply_rotate(input_normals, rotate_dict)

        elif self.coord_system == '3dvc':            

            input_hvc_meta_path = self.hvc_metadata_split_paths[index]
            hvc_meta = np.loadtxt(input_hvc_meta_path)
            hvc_rotate_dict = {'elev': hvc_meta[1], 'azim': hvc_meta[0]}
            input_pointcld = utils.apply_rotate(input_pointcld, hvc_rotate_dict)
            input_normals = utils.apply_rotate(input_normals, hvc_rotate_dict)


            input_metadata_path = self.metadata_split_paths[index]
            meta = np.loadtxt(input_metadata_path)
            rotate_dict = {'elev': meta[img_idx][1], 'azim': meta[img_idx][0]-180}

            input_pointcld = utils.apply_rotate(input_pointcld, rotate_dict)
            input_normals = utils.apply_rotate(input_normals, rotate_dict)

        input_pointcld = torch.FloatTensor(input_pointcld)
        input_normals = torch.FloatTensor(input_normals)

        return input_pointcld, input_normals




    def __getitem__(self, index):
        if self.random_view:
            img_idx = np.random.choice(self.seq_len)
        else:
            img_idx = -1
        image_data = self.get_data_sample(index, img_idx)

        points_data, vals_data = self.get_points_sdf_sample(index, img_idx)
        if self.rep == 'occnet':
            vals_data = (vals_data.cpu().numpy() <= 0.003).astype(np.float32)
            vals_data = torch.FloatTensor(vals_data)

        if self.mode == 'test':
            idx, img_idx = self.get_img_index(index, img_idx)            
            pointcloud_data, normals_data = \
                    self.get_pointcloud_sample(index, img_idx)
            return image_data, points_data, vals_data, pointcloud_data, \
                        normals_data, self.obj_cat_map[idx], img_idx
        return image_data, points_data, vals_data

    def __len__(self):
        if self.image_split_paths != None:
            num_mdl = len(self.image_split_paths)
        elif self.depth_split_paths != None:
            num_mdl = len(self.depth_split_paths)
        elif self.normal_split_paths != None:
            num_mdl = len(self.normal_split_paths)
        else:
            raise Exception("Must have at least 1 input image type")
        if self.random_view:
            return num_mdl
        return num_mdl*self.seq_len

    def get_img_index(self, index, img_idx):
        if img_idx == -1:
            idx = index//self.seq_len
            img_idx = index % self.seq_len
        else:
            idx = index

        return idx, img_idx

        
