import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import sys
import torch.multiprocessing as mp
import subprocess
import config
from datetime import datetime
import utils
from dataloader import Dataset

from model import SDFNet
from tqdm import tqdm
import copy

def main():
    torch.backends.cudnn.benchmark = True

    # log params
    log_dir = config.logging['log_dir']
    exp_name = config.logging['exp_name']
    date = datetime.now().date().strftime("%m_%d_%Y")
    log_dir = os.path.join(log_dir, exp_name, date)
    os.makedirs(log_dir, exist_ok=True)
    utils.writelogfile(log_dir)


    # output directory
    out_dir = config.training['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    batch_size = config.training['batch_size']
    batch_size_eval = config.training['batch_size_eval']
    num_epochs = config.training['num_epochs']

    save_model_step = config.training['save_model_step']
    eval_step = config.training['eval_step']
    verbose_step = config.training['verbose_step']

    # Not doing evaluation on val data
    if eval_step == None:
        eval_step = int(10e9)

    num_points = config.training['num_points']

    cont = config.training['cont']

    rep = config.training['rep']

    coord_system = config.training['coord_system']

    # Dataset
    print('Loading data...')
    train_dataset = Dataset(num_points=num_points, mode='train', rep=rep, \
        coord_system=coord_system)
    eval_train_dataset = Dataset(mode='train', rep=rep, \
        coord_system=coord_system)
    val_dataset = Dataset(mode='val', rep=rep, coord_system=coord_system)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=12, shuffle=True,\
            pin_memory=True)
    eval_train_loader = torch.utils.data.DataLoader(
        eval_train_dataset, batch_size=batch_size_eval, num_workers=12, \
            drop_last=True,pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size_eval, num_workers=12, \
            drop_last=True,pin_memory=True)

    # Model
    print('Initializing network...')
    model = SDFNet()

    # Initialize training
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if rep == 'occ':
        criterion = nn.BCEWithLogitsLoss()
    elif rep == 'sdf':
        criterion = utils.LpLoss
    else:
        raise Exception('Representation not supported, either occ or sdf')

    # Initialize training process
    epoch_it = 0

    if rep == 'occ':
        max_metric_val = 0
    elif rep == 'sdf':
        max_metric_val = np.zeros(2, dtype=np.float32)
    metric_val_array = []
    epoch_val_array = []
    loss_val_array = []

    if rep == 'occ':
        max_metric_train = 0
    elif rep == 'sdf':
        max_metric_train = np.zeros(2, dtype=np.float32)
    metric_train_array = []
    epoch_train_array = []
    loss_train_array = []

    # Resume/continue training
    if cont is not None:
        checkpoint = torch.load(os.path.join(out_dir, cont))
        if not os.path.exists(checkpoint):
            raise Exception('Checkpoint does not exist')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch_it = checkpoint['epoch']

        if os.path.exists(os.path.join(out_dir, 'train.npz')):
            # Load saved data
            try:
                train_npz = np.load(os.path.join(out_dir, 'train.npz'), allow_pickle=True)
                metric_train_array = train_npz['metric']
                epoch_train_array = train_npz['epoch']
                loss_train_array = train_npz['loss']
                metric_train_array = list(metric_train_array[\
                        epoch_train_array <= epoch_it])
                loss_train_array = list(loss_train_array[\
                        epoch_train_array <= epoch_it])
                epoch_train_array = list(epoch_train_array[\
                        epoch_train_array <= epoch_it])

                max_metric_train = np.max(np.asarray(metric_train_array),\
                    axis=0)
            except Exception:
                print('Cannot load train npz')
        if os.path.exists(os.path.join(out_dir, 'val.npz')):
            try:
                val_npz = np.load(os.path.join(out_dir, 'val.npz'), allow_pickle=True)

                metric_val_array = val_npz['metric']
                epoch_val_array = val_npz['epoch']
                loss_val_array = val_npz['loss']

                metric_val_array = list(metric_val_array[\
                        epoch_val_array <= epoch_it])
                loss_val_array = list(loss_val_array[\
                        epoch_val_array <= epoch_it])
                epoch_val_array = list(epoch_val_array[\
                        epoch_val_array <= epoch_it])

                max_metric_val = np.max(np.asarray(metric_val_array),axis=0)
            except Exception:
                print('Cannot load val npz')


    # Saving meta config
    meta_config_path = os.path.join(out_dir, 'meta_config.npz')
    np.savez(meta_config_path, training=config.training, \
        testing=config.testing, data_setting=config.data_setting, \
        logging=config.logging, path=config.path)


    # Data parallel
    model = torch.nn.DataParallel(model).cuda()

    print('Start training...')
    while True:
        epoch_it += 1
        if num_epochs is not None and epoch_it > num_epochs:
            break
        print('Starting epoch %s'%(epoch_it))
        model = train(model, criterion, optimizer, train_loader, batch_size,\
             epoch_it, rep)

        if epoch_it % save_model_step == 0:
            print('Saving model...')
            torch.save({
                'epoch': epoch_it,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},\
                    os.path.join(out_dir, 'model-%s.pth.tar'%(epoch_it)))

        if epoch_it % verbose_step == 0:
            print('Evaluating on train data...')
            mean_loss, mean_metric = eval(model, criterion, optimizer, eval_train_loader, batch_size, epoch_it, rep)
            print('Mean loss on train set: %.4f'%(mean_loss))
            if rep == 'occ':
                print('Mean IoU on train set: %.4f'%(mean_metric[0]))
            elif rep == 'sdf':
                print('Mean IoU on train set: %.4f'%(mean_metric[2]))
                print('Mean accuracy on train set: %.4f'%(mean_metric[1]))

            metric_train_array.append(mean_metric)
            epoch_train_array.append(epoch_it)
            loss_train_array.append(mean_loss)

            np.savez(os.path.join(out_dir, 'train.npz'), metric=metric_train_array, epoch=epoch_train_array, \
                loss=loss_train_array)

            # Saving best model based on train metric
            if rep == 'occ':
                if mean_metric[0] > max_metric_train:
                    max_metric_train = copy.deepcopy(mean_metric[0])
                    torch.save(model.module.state_dict(), \
                        os.path.join(out_dir, 'best_model_train.pth.tar'))
            elif rep == 'sdf':
                if mean_metric[2] > max_metric_train[0]:
                    max_metric_train[0] = copy.deepcopy(mean_metric[2])
                    torch.save(model.module.state_dict(), \
                        os.path.join(out_dir, 'best_model_iou_train.pth.tar'))
                if mean_metric[1] > max_metric_train[1]:
                    max_metric_train[1] = copy.deepcopy(mean_metric[1])
                    torch.save(model.module.state_dict(), \
                        os.path.join(out_dir, 'best_model_acc_train.pth.tar'))

        if epoch_it % eval_step == 0:
            print('Evaluating on test data...')
            mean_loss_val, mean_metric_val = eval(model, criterion, \
                optimizer, val_loader, batch_size, epoch_it, rep)
            print('Mean loss on val set: %.4f'%(mean_loss_val))
            if rep == 'occ':
                print('Mean IoU on val set: %.4f'%(mean_metric_val[0]))
            elif rep == 'sdf':
                print('Mean IoU on val set: %.4f'%(mean_metric_val[2]))
                print('Mean accuracy on val set: %.4f'%(mean_metric_val[1]))

            metric_val_array.append(mean_metric_val)
            epoch_val_array.append(epoch_it)
            loss_val_array.append(mean_loss_val)

            np.savez(os.path.join(out_dir, 'val.npz'), \
                metric=metric_val_array, epoch=epoch_val_array, \
                loss=loss_val_array)

            # Saving best model based on val metric
            if rep == 'occ':
                if mean_metric_val[0] > max_metric_val:
                    max_metric_val = copy.deepcopy(mean_metric_val[0])
                    if cont is None:
                        torch.save(model.module.state_dict(), \
                            os.path.join(out_dir, 'best_model.pth.tar'))
                    else:
                        torch.save(model.module.state_dict(), \
                            os.path.join(out_dir, 'best_model_cont.pth.tar'))
            elif rep == 'sdf':
                if mean_metric_val[2] > max_metric_val[0]:
                    max_metric_val[0] = copy.deepcopy(mean_metric_val[2])
                    if cont is None:
                        torch.save(model.module.state_dict(), \
                            os.path.join(out_dir, 'best_model_iou.pth.tar'))
                    else:
                        torch.save(model.module.state_dict(), \
                            os.path.join(out_dir, \
                                'best_model_iou_cont.pth.tar'))

                if mean_metric_val[1] > max_metric_val[1]:
                    max_metric_val[1] = copy.deepcopy(mean_metric_val[1])
                    if cont is None:
                        torch.save(model.module.state_dict(), \
                            os.path.join(out_dir, 'best_model_acc.pth.tar'))
                    else:
                        torch.save(model.module.state_dict(), \
                            os.path.join(out_dir, \
                                'best_model_acc_cont.pth.tar'))

            del mean_loss_val
    
def train(model, criterion, optimizer, train_loader, \
            batch_size, epoch_it, rep):
    model.train()
    with tqdm(total=int(len(train_loader)), ascii=True) as pbar:
        for mbatch in train_loader:
            img_input, points_input, values = mbatch
            img_input = Variable(img_input).cuda()

            points_input = Variable(points_input).cuda()
            values = Variable(values).cuda()

            optimizer.zero_grad()
            
            logits = model(points_input, img_input)
            if rep == 'occ':
                loss = criterion(logits, values)
            elif rep == 'sdf':
                loss = criterion(logits, values)

            loss.backward()
            optimizer.step()

            del loss
            
            pbar.update(1)
    return model


def eval(model, criterion, optimizer, loader, batch_size, epoch_it, rep):
    model.eval()
    loss_collect = []
    metric_collect = []
    if rep == 'occ':
        sigmoid = torch.nn.Sigmoid()

    with tqdm(total=int(len(loader)), ascii=True) as pbar:
        with torch.no_grad():
            for mbatch in loader:
                img_input, points_input, values = mbatch
                img_input = Variable(img_input).cuda()

                points_input = Variable(points_input).cuda()
                values = Variable(values).cuda()

                optimizer.zero_grad()

                logits = model(points_input, img_input)

                loss = criterion(logits, values)

                loss_collect.append(loss.data.cpu().item())

                if rep == 'occ':
                    logits = sigmoid(logits)

                    iou = utils.compute_iou(logits.detach().cpu().numpy(), \
                                values.detach().cpu().numpy())
                    metric_collect.append(iou)
                elif rep == 'sdf':
                    # acc_sign is sign IoU
                    # acc_thres is accuracy within a threshold
                    # More detail explanation in utils.py
                    acc_sign, acc_thres, iou = utils.compute_acc(\
                                        logits.detach().cpu().numpy(), \
                                        values.detach().cpu().numpy())
                    metric_collect.append([acc_sign, acc_thres, iou])
                pbar.update(1)

    mean_loss = np.mean(np.array(loss_collect))
    mean_metric = np.mean(np.array(metric_collect), axis=0)

    return mean_loss, mean_metric

if __name__ == '__main__':
    main()










