# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------
# import sys
# print("PATH")
# print(sys.path)
# sys.path.insert(0, "/Users/wuzhelun/Code/DenseFusion_0.4/lib/knn/knn_pytorch")
# print(sys.path)

import _init_paths
import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset as PoseDataset_ycb
from datasets.linemod.dataset import PoseDataset as PoseDataset_linemod
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
# from lib.loss_refiner import Loss_refine
from lib.utils import setup_logger
from datasets.ycb.dataset import get_bbox as get_bbox
from PIL import Image
import scipy.io as scio
import numpy.ma as ma
from numpy import linalg as LA
from scipy.optimize import fmin
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
# from sympy import Plane, Point3D


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

PRED_ON_PLANE_FACTOR = 0.5
PRODUCT_THRESHOLD = math.cos(math.pi / 180 * 20)

def generate_obj_file(sym_list, center, cloud, count):

    thresh = 2e-3
    with open('gen_obj/labeled_{}.obj'.format(count), 'w') as f:
        center = center.unsqueeze(1).repeat(1,cloud.size(1),1)
        cloud_shifted = cloud - center
        sym_list = sym_list.permute(0,2,1)
        dist = torch.bmm(cloud_shifted, sym_list)
        dist = torch.abs(dist)
        how_close, which_close = torch.min(dist, 2)

        
        for i in range(cloud.size(1)):
            if how_close[0,i] < thresh:
                f.write("v {} {} {} {} {} {}\n".format(cloud[0,i,0], cloud[0,i,1], cloud[0,i,2], 255*(1+sym_list[0,0,which_close[0,i]])/2, 255*(1+sym_list[0,1,which_close[0,i]])/2, 255*(1+sym_list[0,2,which_close[0,i]])/2))
            else:
                f.write("v {} {} {} {} {} {}\n".format(cloud[0,i,0], cloud[0,i,1], cloud[0,i,2], 0, 0, 0))

def generate_obj_file_norm_pred(pred_norm, pred_on_plane, cloud, idx1, idx2, idx3):

    with open('gen_obj_norm_pred/labeled_scene_{}_{}_idx_{}.obj'.format(idx1, idx2, idx3), 'w') as f:
        for i in range(cloud.size(1)):
            if pred_on_plane[:,i,:] > max(0.5,pred_on_plane.max()*PRED_ON_PLANE_FACTOR + pred_on_plane.mean() * (1-PRED_ON_PLANE_FACTOR)):
                f.write("v {} {} {} {} {} {}\n".format(cloud[0,i,0], cloud[0,i,1], cloud[0,i,2], 255*(1+pred_norm[0,i,0].item())/2, 255*(1+pred_norm[0,i,1].item())/2, 255*(1+pred_norm[0,i,2].item())/2))
            else:
                f.write("v {} {} {} {} {} {}\n".format(cloud[0,i,0], cloud[0,i,1], cloud[0,i,2], 0, 0, 0))

def generate_obj_file_norm(pred_norm, cloud, idx1, idx2, idx3, count):

    # import pdb;pdb.set_trace()
    with open('gen_obj_norm_scrub/labeled_scene_{}_{}_idx_{}_scrub_{}.obj'.format(idx1, idx2, idx3, count), 'w') as f:
        for i in range(cloud.shape[0]):
            f.write("v {} {} {} {} {} {}\n".format(cloud[i,0], cloud[i,1], cloud[i,2], 255*(1+pred_norm[i,0])/2, 255*(1+pred_norm[i,1])/2, 255*(1+pred_norm[i,2])/2))

def generate_obj_file_sym_pred(norm, center, cloud, idx1, idx2, idx3, idx4):

    thresh = 2e-3

    # import pdb;pdb.set_trace()

    norm = torch.from_numpy(norm.astype(np.float32))
    center = torch.from_numpy(center.astype(np.float32))
    cloud = torch.tensor(cloud)
    cloud = cloud.squeeze()
    
    with open('gen_obj_sym_pred/labeled_scene_{}_{}_idx_{}_sym_{}.obj'.format(idx1, idx2,idx3,idx4), 'w') as f:
        center = center.unsqueeze(0).repeat(cloud.size(0),1)
        cloud_shifted = cloud - center
        # cloud_shifted = F.normalize(cloud_shifted,dim=1)
        
        # norm = norm.permute(1,0)
        # print(cloud_shifted.size())
        # print(norm.size())
        dist = torch.mm(cloud_shifted, norm)
        dist = torch.abs(dist).squeeze()

        torch.set_printoptions(threshold=5000)
        for i in range(cloud.size(0)):
            if dist[i] < thresh:
                f.write("v {} {} {} {} {} {}\n".format(cloud[i,0], cloud[i,1], cloud[i,2], 255, 255, 255))
            else:
                f.write("v {} {} {} {} {} {}\n".format(cloud[i,0], cloud[i,1], cloud[i,2], 0, 0, 0))


def main():
    # opt.manualSeed = random.randint(1, 10000)
    # # opt.manualSeed = 1
    # random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed)

    torch.set_printoptions(threshold=5000)
    # device_ids = [0,1]
    cudnn.benchmark = True
    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 3 #number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return

    estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)

    estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    # refiner.cuda()
    # estimator = nn.DataParallel(estimator, device_ids=device_ids)
    
    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))
        print('LOADED!!')

    if opt.resume_refinenet != '':
        refiner.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_refinenet)))
        opt.refine_start = True
        opt.decay_start = True
        opt.lr *= opt.lr_rate
        opt.w *= opt.w_rate
        opt.batch_size = int(opt.batch_size / opt.iteration)
        optimizer = optim.Adam(refiner.parameters(), lr=opt.lr)
    else:
        print('no refinement')
        opt.refine_start = False
        opt.decay_start = False
        optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
        # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        
    if opt.dataset == 'ycb':
        dataset = PoseDataset_ycb('train', opt.num_points, False, opt.dataset_root, opt.noise_trans, opt.refine_start)
        # print(dataset.list)
    elif opt.dataset == 'linemod':
        dataset = PoseDataset_linemod('train', opt.num_points, True, opt.dataset_root, opt.noise_trans, opt.refine_start)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=opt.workers)
    if opt.dataset == 'ycb':
        test_dataset = PoseDataset_ycb('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    elif opt.dataset == 'linemod':
        test_dataset = PoseDataset_linemod('test', opt.num_points, False, opt.dataset_root, 0.0, opt.refine_start)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.workers)
    
    opt.sym_list = dataset.get_sym_list()
    opt.num_points_mesh = dataset.get_num_points_mesh()

    # print('>>>>>>>>----------Dataset loaded!---------<<<<<<<<\nlength of the training set: {0}\nlength of the testing set: {1}\nnumber of sample points on mesh: {2}\nsymmetry object list: {3}'.format(len(dataset), len(test_dataset), opt.num_points_mesh, opt.sym_list))

    criterion = Loss(opt.num_points_mesh, opt.sym_list)
    # criterion_refine = Loss_refine(opt.num_points_mesh, opt.sym_list)

    best_test = np.Inf
    best_epoch = 0

    if opt.start_epoch == 1:
        for log in os.listdir(opt.log_dir):
            os.remove(os.path.join(opt.log_dir, log))
    st_time = time.time()

    count_gen = 0

    mode = 1

    if mode == 1:

        for epoch in range(opt.start_epoch, opt.nepoch):
            logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
            logger.info('Train time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
            train_count = 0
            train_dis_avg = 0.0
            if opt.refine_start:
                estimator.eval()
                refiner.train()
            else:
                estimator.train()
            optimizer.zero_grad()

            for rep in range(opt.repeat_epoch):
                for i, data in enumerate(dataloader, 0):
                    points, choose, img, target_sym, target_cen, idx, file_list_idx = data

                    if idx is 9 or idx is 16:
                        continue

                    points, choose, img, target_sym, target_cen, idx = Variable(points).cuda(), \
                                                                     Variable(choose).cuda(), \
                                                                     Variable(img).cuda(), \
                                                                     Variable(target_sym).cuda(), \
                                                                     Variable(target_cen).cuda(), \
                                                                     Variable(idx).cuda()

                    pred_norm, pred_on_plane, emb = estimator(img, points, choose, idx)

                    loss = criterion(pred_norm, pred_on_plane, target_sym, target_cen, idx, points, opt.w, opt.refine_start)

                    # scene_idx = dataset.list[file_list_idx]

                    loss.backward()

                    # train_dis_avg += dis.item()
                    train_count += 1

                    if train_count % opt.batch_size == 0:
                        logger.info('Train time {0} Epoch {1} Batch {2} Frame {3}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, int(train_count / opt.batch_size), train_count))
                        optimizer.step()
                        # for param_lr in optimizer.module.param_groups:
                        #         param_lr['lr'] /= 2
                        optimizer.zero_grad()
                        train_dis_avg = 0

                    if train_count % 8 == 0:
                        print(pred_on_plane.max())
                        print(pred_on_plane.mean())
                        print(idx)

                    if train_count != 0 and train_count % 1000 == 0:
                        if opt.refine_start:
                            torch.save(refiner.state_dict(), '{0}/pose_refine_model_current.pth'.format(opt.outf))
                        else:
                            torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))

            print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))

            logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
            logger.info('Test time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
            test_loss = 0.0
            test_count = 0
            estimator.eval()

            logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch, test_loss))
            print(pred_on_plane.max())
            print(pred_on_plane.mean())
            bs, num_p, _ = pred_on_plane.size()
            # if epoch % 40 == 0:
            #     import pdb;pdb.set_trace()
            best_test = test_loss
            best_epoch = epoch
            if opt.refine_start:
                torch.save(refiner.state_dict(), '{0}/pose_refine_model_{1}_{2}.pth'.format(opt.outf, epoch, test_loss))
            else:
                torch.save(estimator.state_dict(), '{0}/pose_model_{1}_{2}.pth'.format(opt.outf, epoch, test_loss))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')

            if best_test < opt.decay_margin and not opt.decay_start:
                opt.decay_start = True
                opt.lr *= opt.lr_rate
                # opt.w *= opt.w_rate
                optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
        
        estimator.load_state_dict(torch.load('{0}/pose_model_{1}_{2}.pth'.format(opt.outf, best_epoch, best_test)))
    else:
        estimator.load_state_dict(torch.load('{0}/pose_model_45_0.0.pth'.format(opt.outf), map_location='cpu'))


if __name__ == '__main__':
    main()

