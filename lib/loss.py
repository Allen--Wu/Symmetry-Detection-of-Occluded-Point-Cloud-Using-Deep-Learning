from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import time
import numpy as np
import torch.nn as nn
import random
# import torch.backends.cudnn as cudnn
# from lib.knn.__init__ import KNearestNeighbor


def loss_calculation(pred_norm, pred_on_plane, target_sym, target_cen, idx, points, w, refine, num_point_mesh, sym_list):
    # knn = KNearestNeighbor(1)
    bs, num_p, _ = pred_on_plane.size()
    thresh = 2e-3
    broad_thresh = 3e-3
    # sym = target_sym
    pred_norm = pred_norm / (torch.norm(pred_norm, dim=2).view(bs, num_p, 1))
    # pred_norm = pred_norm.repeat(1,target_sym.size(1),1)
    # target_sym = target_sym.repeat(1,1000,1)
    # loss_norm = torch.abs(target_sym-pred_norm).mean()
    target_cen = target_cen.unsqueeze(1).repeat(1,points.size(1),1)
    points_shifted = points - target_cen
    target_sym = target_sym.permute(0,2,1)
    dist = torch.bmm(points_shifted, target_sym)
    dist = torch.abs(dist)
    how_close, which_close = torch.min(dist, 2)

    loss = 0
    # import pdb;pdb.set_trace()
    if pred_on_plane.max() > 0.6:

        close_index = how_close[0,:] < broad_thresh
        if close_index.sum() > 0:
            loss += 10*torch.norm(pred_norm[0,close_index,:] - target_sym[0,:,which_close[0,close_index]].permute(1,0), p=1, dim=1).sum()-torch.log(pred_on_plane[0,close_index,0]).sum()

        far_index = how_close[0,:] >= broad_thresh
        if far_index.sum() > 0:
            loss += -torch.log(1-pred_on_plane[:,far_index,:]).squeeze().sum()
    else:
        
        close_index = how_close[0,:] < thresh
        if close_index.sum() > 0:
            loss += -torch.log(pred_on_plane[:,close_index,:]).squeeze().sum()

        far_index = how_close[0,:] >= thresh
        if far_index.sum() > 0:
            loss += -torch.log(1-pred_on_plane[:,far_index,:]).squeeze().sum()

    # for i in range(points.size(1)):
    #     if on_plane_good:
    #         if how_close[:,i] < broad_thresh:
    #             # import pdb;pdb.set_trace()
    #             loss += 10*torch.norm(pred_norm[:,i,:].squeeze() - target_sym[:,:,which_close[:,i]].squeeze(), p=1, dim=0)-torch.log(pred_on_plane[:,i,:]).squeeze()
    #         else:
    #             loss += -torch.log(1-pred_on_plane[:,i,:]).squeeze()
    #     else:
    #         if how_close[:,i] < thresh:
    #             loss += -torch.log(pred_on_plane[:,i,:]).squeeze()
    #         else:
    #             loss += -torch.log(1-pred_on_plane[:,i,:]).squeeze()

    loss /= points.size(1)
    # print(loss)
    return loss


class Loss(_Loss):

    def __init__(self, num_points_mesh, sym_list):
        super(Loss, self).__init__(True)
        self.num_pt_mesh = num_points_mesh
        self.sym_list = sym_list

    def forward(self, pred_norm, pred_on_plane, target_sym, target_cen, idx, points, w, refine):

        return loss_calculation(pred_norm, pred_on_plane, target_sym, target_cen, idx, points, w, refine, self.num_pt_mesh, self.sym_list)
