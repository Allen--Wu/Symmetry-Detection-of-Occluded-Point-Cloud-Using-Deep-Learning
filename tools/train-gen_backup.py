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

    # estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    # refiner.cuda()
    # estimator = nn.DataParallel(estimator, device_ids=device_ids)
    
    if opt.resume_posenet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_posenet)))

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

    mode = 0

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

                    # points, choose, img, target_sym, target_cen, target, idx, file_list_idx = data
                    # generate_obj_file(target_sym, target_cen, target, idx.squeeze())
                    # import pdb;pdb.set_trace()
                    points, choose, img, target_sym, target_cen, idx = Variable(points).cuda(), \
                                                                     Variable(choose).cuda(), \
                                                                     Variable(img).cuda(), \
                                                                     Variable(target_sym).cuda(), \
                                                                     Variable(target_cen).cuda(), \
                                                                     Variable(idx).cuda()
                    # points, choose, img, target_sym, target_cen, idx = Variable(points), \
                    #                                                 Variable(choose), \
                    #                                                 Variable(img), \
                    #                                                 Variable(target_sym), \
                    #                                                 Variable(target_cen), \
                    #                                                 Variable(idx)
                    pred_norm, pred_on_plane, emb = estimator(img, points, choose, idx)

                    # pred_norm_new = torch.cat((pred_norm, torch.zeros(1,pred_norm.size(1),1)),2)

                    # for i in range(pred_norm.size(1)):
                    #     pred_norm_new[0,i,2] = torch.sqrt(1 - pred_norm[0,i,0] * pred_norm[0,i,0] - pred_norm[0,i,1] * pred_norm[0,i,1])                    
                    # if epoch % 10 == 0:
                    #     generate_obj_file_pred(pred_norm, pred_on_plane, points, count_gen, idx)
                    #     count_gen += 1
                    # print(pred_norm[0,0,:])

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

                    if train_count % 5000 == 0:
                        print(pred_on_plane.max())
                        print(pred_on_plane.mean())

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
            # refiner.eval()

            # for rep in range(opt.repeat_epoch):
            #     for j, data in enumerate(testdataloader, 0):
            #         points, choose, img, target_sym, target_cen, idx, img_idx = data
            #         # points, choose, img, target, model_points, idx = Variable(points).cuda(), \
            #         #                                                  Variable(choose).cuda(), \
            #         #                                                  Variable(img).cuda(), \
            #         #                                                  Variable(target).cuda(), \
            #         #                                                  Variable(model_points).cuda(), \
            #         #                                                  Variable(idx).cuda()
            #         points, choose, img, target_sym, target_cen, idx = Variable(points), \
            #                                                             Variable(choose), \
            #                                                             Variable(img), \
            #                                                             Variable(target_sym), \
            #                                                             Variable(target_cen), \
            #                                                             Variable(idx)

            #         pred_norm, pred_on_plane, emb = estimator(img, points, choose, idx)
            #         loss = criterion(pred_norm, pred_on_plane, target_sym, target_cen, idx, points, opt.w, opt.refine_start)
            #         test_loss += loss

            #         logger.info('Test time {0} Test Frame No.{1}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), test_count))

            #         test_count += 1

            # test_loss = test_loss / test_count
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


    
    for DIST_THRESHOLD in np.arange(0.01, 0.06, 0.01):

        for take_idx in range(3):

            conf_tp_or_fn = []
            conf_fp = []

            true_positives = 0
            false_positives = 0
            false_negatives = 0
            if take_idx is 0:
                take_into_account = {1:[2],2:[1],3:[0],4:[0,1],6:[1],7:[0,1],8:[0,1,2],11:[0],13:[0],15:[1,2],20:[0]}
            elif take_idx is 1:
                take_into_account = {0:[0],2:[0],6:[0],10:[0],12:[0],18:[0]}
            else:
                take_into_account = {1:[0,1],15:[0],19:[0]}

            for index in range(len(test_dataset.list)):
                img = Image.open('{0}/data_v1/{1}-color.png'.format(test_dataset.root, test_dataset.list[index]))
                depth = np.array(Image.open('{0}/data_v1/{1}-depth.png'.format(test_dataset.root, test_dataset.list[index])))
                label = np.array(Image.open('{0}/data_v1/{1}-label.png'.format(test_dataset.root, test_dataset.list[index])))
                meta = scio.loadmat('{0}/data_v1/{1}-meta.mat'.format(test_dataset.root, test_dataset.list[index]))

                cam_cx = test_dataset.cam_cx_1
                cam_cy = test_dataset.cam_cy_1
                cam_fx = test_dataset.cam_fx_1
                cam_fy = test_dataset.cam_fy_1
                mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

                obj = meta['cls_indexes'].flatten().astype(np.int32)
                for idx in range(0, len(obj)):
                    if not obj[idx] in take_into_account:
                        continue
                    print('scene index: ',test_dataset.list[index])            
                    print('object index: ', obj[idx])
                    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
                    mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
                    mask = mask_label * mask_depth
                    if not (len(mask.nonzero()[0]) > test_dataset.minimum_num_pt and len(test_dataset.symmetry[obj[idx]]['mirror'])>0):
                        continue

                    rmin, rmax, cmin, cmax = get_bbox(mask_label)
                    img_temp = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

                    img_masked = img_temp           
                    target_r = meta['poses'][:, :, idx][:, 0:3]
                    target_t = np.array(meta['poses'][:, :, idx][:, 3:4].flatten())
                    add_t = np.array([random.uniform(-test_dataset.noise_trans, test_dataset.noise_trans) for i in range(3)])

                    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                    if len(choose) > test_dataset.num_pt:
                        c_mask = np.zeros(len(choose), dtype=int)
                        c_mask[:test_dataset.num_pt] = 1
                        np.random.shuffle(c_mask)
                        choose = choose[c_mask.nonzero()]
                    else:
                        choose = np.pad(choose, (0, test_dataset.num_pt - len(choose)), 'wrap')
                    
                    depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    xmap_masked = test_dataset.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    ymap_masked = test_dataset.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
                    choose = np.array([choose])

                    cam_scale = meta['factor_depth'][0][0]
                    pt2 = depth_masked / cam_scale
                    pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
                    pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
                    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

                    dellist = [j for j in range(0, len(test_dataset.cld[obj[idx]]))]

                    # dellist = random.sample(dellist, len(test_dataset.cld[obj[idx]]) - test_dataset.num_pt_mesh_small)
                    # model_points = np.delete(test_dataset.cld[obj[idx]], dellist, axis=0)
                    model_points = test_dataset.cld[obj[idx]]

                    target_sym = []
                    for sym in test_dataset.symmetry[obj[idx]]['mirror']:
                        target_sym.append(np.dot(sym, target_r.T))
                    target_sym = np.array(target_sym)

                    target_cen = np.add(test_dataset.symmetry[obj[idx]]['center'], target_t)

                    target = np.dot(model_points, target_r.T)
                    target = np.add(target, target_t)

                    print('ground truth norm: ', target_sym)
                    print('ground truth center: ', target_cen)
                    points_ten, choose_ten, img_ten, target_sym_ten, target_cen_ten, target_ten, idx_ten = \
                    torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0), \
                    torch.LongTensor(choose.astype(np.int32)).unsqueeze(0), \
                    test_dataset.norm(torch.from_numpy(img_masked.astype(np.float32))).unsqueeze(0), \
                    torch.from_numpy(target_sym.astype(np.float32)).unsqueeze(0), \
                    torch.from_numpy(target_cen.astype(np.float32)).unsqueeze(0), \
                    torch.from_numpy(target.astype(np.float32)).unsqueeze(0), \
                    torch.LongTensor([obj[idx]-1]).unsqueeze(0)
                    
                    # print(img_ten.size())
                    # print(points_ten.size())
                    # print(choose_ten.size())
                    # print(idx_ten.size())

                    points_ten, choose_ten, img_ten, target_sym_ten, target_cen_ten, idx_ten = Variable(points_ten), \
                                                                        Variable(choose_ten), \
                                                                        Variable(img_ten), \
                                                                        Variable(target_sym_ten), \
                                                                        Variable(target_cen_ten), \
                                                                        Variable(idx_ten)

                    pred_norm, pred_on_plane, emb = estimator(img_ten, points_ten, choose_ten, idx_ten)

                    # import pdb;pdb.set_trace() 

                    bs, num_p, _ = pred_on_plane.size()


                    # generate_obj_file_norm_pred(pred_norm / (torch.norm(pred_norm, dim=2).view(bs, num_p, 1)), pred_on_plane, points_ten,test_dataset.list[index].split('/')[0], test_dataset.list[index].split('/')[1], obj[idx])

                    loss = criterion(pred_norm, pred_on_plane, target_sym_ten, target_cen_ten, idx, points_ten, opt.w, opt.refine_start)
                    # print('test loss: ', loss)

                    # bs, num_p, _ = pred_on_plane.size()
                    pred_norm = pred_norm / (torch.norm(pred_norm, dim=2).view(bs, num_p, 1))
                    pred_norm = pred_norm.detach().numpy()
                    pred_on_plane = pred_on_plane.detach().numpy()
                    points = points_ten.detach().numpy()
                
                    clustering_points_idx = np.where(pred_on_plane>max(0.5,pred_on_plane.max()*PRED_ON_PLANE_FACTOR + pred_on_plane.mean() * (1-PRED_ON_PLANE_FACTOR)))[1]
                    clustering_norm = pred_norm[0,clustering_points_idx,:]
                    clustering_points = points[0,clustering_points_idx,:]
                    num_points = len(clustering_points_idx)

                    # import pdb;pdb.set_trace()

                    close_thresh = 2e-3
                    broad_thresh = 3e-3

                    sym_conf = [0.0 for i in range(target_sym.shape[0])]

                    count_pred = 0
                    while True:
                        count_pred += 1
                        if num_points <= 20 or count_pred > 3:
                            break
                        # if count_pred > target_sym.shape[0]:
                        #     break

                        # if count_pred > 3:
                        #     break

                        # generate_obj_file_norm(clustering_norm, clustering_points, test_dataset.list[index].split('/')[0], test_dataset.list[index].split('/')[1], obj[idx], count_pred)

                        best_fit_num = 0

                        count_try = 0

                        # import pdb; pdb.set_trace()
                        for j in range(10):

                            pick_idx = np.random.randint(0,num_points-1)
                            pick_point = clustering_points[pick_idx]
                            # proposal_norm = np.array(Plane(Point3D(pick_points[0]),Point3D(pick_points[1]),Point3D(pick_points[2])).normal_vector).astype(np.float32)
                            proposal_norm = clustering_norm[pick_idx]
                            proposal_norm = proposal_norm[:,np.newaxis]

                            # import pdb;pdb.set_trace()
                            proposal_point = pick_point
                            # highest_pred_idx = np.argmax(pred_on_plane[0,clustering_points_idx,:])
                            # highest_pred_loc = clustering_points[highest_pred_idx]
                            # proposal_norm = clustering_norm[highest_pred_idx][:,np.newaxis]
                            clustering_diff = clustering_points - proposal_point
                            clustering_dist = np.abs(np.matmul(clustering_diff, proposal_norm))

                            broad_inliers = np.where(clustering_dist < broad_thresh)[0]
                            broad_inlier_num = len(broad_inliers)

                            close_inliers = np.where(clustering_dist < close_thresh)[0]
                            close_inlier_num = len(close_inliers)

                            norm_dist = np.abs(clustering_norm-np.transpose(proposal_norm)).sum(1)
                            close_norm_idx = np.where(norm_dist < 0.6)[0]
                            close_norm_num =  len(close_norm_idx)

                            if close_inlier_num >= best_fit_num and broad_inlier_num >= num_points / (4-count_pred) *0.9 and close_norm_num >= num_points / (4-count_pred) * 0.9:
                                best_fit_num = close_inlier_num
                                best_fit_norm = proposal_norm
                                best_fit_cen = clustering_points[close_inliers].mean(0)
                                best_fit_idx = clustering_points_idx[close_inliers]
                                best_norm_dist = norm_dist
                                best_close_norm_idx = np.where(best_norm_dist < 0.6)[0]


                        if best_fit_num == 0 or num_points <= 20:
                            break

                        print('proposal: ',best_fit_norm)
                        print('close inlier: ',best_fit_num)
                        print('num points: ',num_points)
                        print('require: ', num_points / (4-count_pred) * 0.9)
                        # print('distance: ', best_norm_dist)
                        # print('same sym distance: ', best_norm_dist[best_close_norm_idx])

                        # import pdb;pdb.set_trace()
                        clustering_points_same_sym = clustering_points[best_close_norm_idx]

                        clustering_diff_same_sym = clustering_points_same_sym - best_fit_cen
                        clustering_dist_same_sym = np.abs(np.matmul(clustering_diff_same_sym, best_fit_norm))

                        close_inliers = np.where(clustering_dist_same_sym < close_thresh)[0]
                        close_inlier_num= len(close_inliers)

                        best_fit_num = close_inlier_num

                        broad_inliers = np.where(clustering_dist_same_sym < broad_thresh)[0]
                        broad_inlier_num= len(broad_inliers)

                        def f(x):
                            dist = 0
                            # import pdb;pdb.set_trace()
                            for point in clustering_points_same_sym[broad_inliers]:
                                dist += np.abs((point * x[0:3]).sum() + x[3]) / np.sqrt(np.sum(np.square(x[0:3]), axis=0))

                            return dist

                        start_point = np.zeros(4)
                        start_point[0:3] = np.copy(best_fit_norm[:,0])
                        start_point[3] = (-best_fit_cen*best_fit_norm[:,0]).sum()

                        min_point = fmin(f, start_point, maxiter=50)

                        # import pdb;pdb.set_trace()
                        min_point = min_point / np.sqrt(np.sum(np.square(min_point[0:3]), axis=0))

                        x_val = -(min_point[3]+best_fit_cen[1] * min_point[1] + best_fit_cen[2] * min_point[2]) / min_point[0]

                        y_val = -(min_point[3]+best_fit_cen[0] * min_point[0] + best_fit_cen[2] * min_point[2]) / min_point[1]

                        z_val = -(min_point[3]+best_fit_cen[0] * min_point[0] + best_fit_cen[1] * min_point[1]) / min_point[2]

                        if np.abs(x_val) < 1:
                            new_pred_loc = np.array([x_val, best_fit_cen[1], best_fit_cen[2]])
                        elif np.abs(z_val) < 1:
                            new_pred_loc = np.array([best_fit_cen[0], best_fit_cen[1], z_val])
                        else:
                            new_pred_loc = np.array([best_fit_cen[0], y_val, best_fit_cen[2]])


                        new_proposal_norm = min_point[0:3]
                        clustering_diff = clustering_points_same_sym - new_pred_loc
                        clustering_dist = np.abs(np.matmul(clustering_diff, new_proposal_norm))

                        close_inliers = np.where(clustering_dist < close_thresh)[0]
                        new_close_inlier_num = len(close_inliers)

                        broad_inliers = np.where(clustering_dist < broad_thresh)[0]
                        new_broad_inlier_num = len(broad_inliers)
                        # import pdb;pdb.set_trace()
                        if new_close_inlier_num >= close_inlier_num:
                            best_fit_num = new_close_inlier_num
                            best_fit_norm = new_proposal_norm[:,np.newaxis]
                            best_fit_cen = new_pred_loc

                        if best_fit_num == 0:
                            break
                        else:
                            print('predicted norm:{}, predicted point:{}'.format(best_fit_norm, best_fit_cen))

                            max_idx = np.argmax(np.abs(np.matmul(target_sym, best_fit_norm)))
                            sym_product = np.abs(np.matmul(target_sym, best_fit_norm)[max_idx][0])
                            sym_dist = np.abs((target_sym[max_idx] * (best_fit_cen-target_cen)).sum())

                            norm_dist = np.abs(clustering_norm-np.transpose(best_fit_norm)).sum(1)
                            scrub_close_norm_idx = np.where(norm_dist < 1.3)[0]

                            # import pdb;pdb.set_trace()
                            predicted_confidence = best_fit_num / len(best_close_norm_idx) - np.abs(clustering_norm[best_close_norm_idx]-np.transpose(best_fit_norm)).mean() * 3 * 1.5
                            predicted_confidence = max(0, predicted_confidence)

                            if sym_product > PRODUCT_THRESHOLD and sym_dist < DIST_THRESHOLD:
                                sym_conf[max_idx] = max(sym_conf[max_idx], predicted_confidence)
                            elif max_idx in take_into_account[obj[idx]]:
                                conf_fp.append(predicted_confidence)

                            print('confidence: ', predicted_confidence)

                            # generate_obj_file_sym_pred(best_fit_norm, best_fit_cen, target_ten, test_dataset.list[index].split('/')[0], test_dataset.list[index].split('/')[1], obj[idx], count_pred)
                            clustering_points_idx = np.setdiff1d(clustering_points_idx, clustering_points_idx[scrub_close_norm_idx])
                            
                            clustering_norm = pred_norm[0,clustering_points_idx,:]
                            clustering_points = points[0,clustering_points_idx,:]
                            
                            print('scrubbed distance: ', np.abs(clustering_norm-np.transpose(best_fit_norm)).sum(1))
                            num_points = len(clustering_points_idx)

                    # import pdb;pdb.set_trace()
                    
                    for i in range(target_sym.shape[0]):
                        if i in take_into_account[obj[idx]]:
                            conf_tp_or_fn.append(sym_conf[i])

            conf_tp_or_fn = np.array(conf_tp_or_fn)
            conf_fp = np.array(conf_fp)

            # import pdb;pdb.set_trace()
            
            prec = []
            recall = []
            for t in range(1,10001):
                conf_thresh = t/10000
                
                true_positives = len(np.where(conf_tp_or_fn >= conf_thresh)[0])
                false_negatives = len(np.where(conf_tp_or_fn < conf_thresh)[0])
                false_positives = len(np.where(conf_fp >= conf_thresh)[0])

                if false_positives + true_positives > 0 and true_positives + false_negatives > 0:
                    prec.append(true_positives / (false_positives + true_positives))
                    recall.append(true_positives / (true_positives + false_negatives))

                # if t % 20 == 0:
                #     import pdb;pdb.set_trace()
            

            # print(prec)
            # print(recall)
            plt.plot(recall, prec)
            plt.axis([0, 1, 0, 1])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.savefig('prec-recall-{}-{}-{}.png'.format(DIST_THRESHOLD, (take_idx+1)*5, (take_idx+2)*5))
            plt.clf()

if __name__ == '__main__':
    main()

