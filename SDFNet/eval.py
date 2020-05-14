import torch
import numpy as np
import os
import trimesh
from tqdm import tqdm
import config
from dataloader import Dataset

from model import SDFNet
from torch.autograd import Variable
import torch.optim as optim
import utils


def main():
    torch.backends.cudnn.benchmark = True

    out_dir = config.training['out_dir']
    rep = config.training['rep']

    model_selection_path = config.testing['model_selection_path']
    cont = config.training['cont']

    if model_selection_path is not None:
        # Model selection path is specified
        model_selection_path = os.path.join(out_dir, model_selection_path)
        model_selection = np.load(model_selection_path, allow_pickle=True)
        ep = model_selection['epoch']
        model_path = 'model-%s.pth.tar'%(ep)
        model_path = os.path.join(out_dir, model_path)
    else:
        if rep == 'occ':
            if cont is None:
                model_path = os.path.join(out_dir, 'best_model.pth.tar')
            else:
                model_path = os.path.join(out_dir, 'best_model_cont.pth.tar')
        elif rep == 'sdf':
            if cont is None:
                model_path = os.path.join(out_dir, 'best_model_iou.pth.tar')
            else:
                model_path = os.path.join(out_dir, \
                    'best_model_iou_cont.pth.tar')

    print('Loading model from %s'%(model_path))

    eval_task_name = config.testing['eval_task_name']
    eval_dir = os.path.join(out_dir, 'eval')
    eval_task_dir = os.path.join(eval_dir, eval_task_name)
    os.makedirs(eval_task_dir, exist_ok=True)

    batch_size_test = config.testing['batch_size_test']
    coord_system = config.training['coord_system']

    box_size = config.testing['box_size']

    # Dataset
    print('Loading data...')
    test_dataset = Dataset(mode='test', rep=rep, coord_system=coord_system)

    test_loader = torch.utils.data.DataLoader( \
        test_dataset, batch_size=batch_size_test, \
            num_workers=12, pin_memory=True)

    # Loading model
    model = SDFNet()
    if model_selection_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(model_path))
    model = torch.nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.eval()
    out_obj_cat = []
    out_pose = []
    out_cd = []
    out_normals = []
    out_iou = []
    out_fscore = []

    with tqdm(total=int(len(test_loader)), ascii=True) as pbar:
        with torch.no_grad():
            for mbatch in test_loader:
                img_input, points_input, values, pointclouds, normals, \
                    obj_cat, pose = mbatch
                img_input = Variable(img_input).cuda()

                points_input = Variable(points_input).cuda()
                values = Variable(values).cuda()

                optimizer.zero_grad()

                obj, cat = obj_cat
                cat_path = os.path.join(eval_task_dir, cat[0])

                os.makedirs(cat_path, exist_ok=True)
                if rep == 'occ':
                    mesh = utils.generate_mesh(img_input, points_input, \
                        model.module)
                    obj_path = os.path.join(cat_path, '%s.off' % obj[0])
                    mesh.export(obj_path)
                elif rep == 'sdf':
                    obj_path = os.path.join(cat_path, '%s.obj' % obj[0])
                    sdf_path = os.path.join(cat_path, '%s.dist' % obj[0])
                    utils.generate_mesh_sdf(img_input, model.module, \
                        obj_path, sdf_path, box_size=box_size)

                # Save gen info
                out_obj_cat.append(obj_cat)
                out_pose.append(pose)

                # Calculate metrics
                if rep == 'occ':
                    out_dict = utils.eval_mesh(mesh, pointclouds, normals,\
                                points_input, values)
                elif rep == 'sdf':
                    if os.path.exists(obj_path):
                        #### Load mesh
                        mesh = trimesh.load(obj_path)
                    else:
                        mesh = None
                    sdf_val = model(points_input, img_input)
                    out_dict = utils.eval_mesh(mesh, pointclouds, normals, \
                                points_input, values, rep='sdf',\
                                sdf_val=sdf_val)
                
                out_cd.append(out_dict['cd'])
                out_normals.append(out_dict['normals'])
                out_iou.append(out_dict['iou'])
                out_fscore.append(out_dict['fscore'])
                np.savez(os.path.join(eval_task_dir, 'out.npz'), \
                    obj_cat=np.array(out_obj_cat), pose=np.array(out_pose),\
                    cd=np.array(out_cd), normals=np.array(out_normals),\
                    iou=np.array(out_iou), fscore=np.array(out_fscore))
                
                pbar.update(1)

if __name__ == '__main__':
    main()



