# Symmetry Detection of Occluded Point Cloud Using Deep Learning
> Zhelun Wu, Hongyan Jiang, Siyun He

> https://arxiv.org/pdf/2003.06520


## Overview
Symmetry detection has been a classical problem in computer graphics, many of which using traditional geometric methods. In recent years, however, we have witnessed the arising deep learning changed the landscape of computer graphics. In this paper, we aim to solve the symmetry detection of the occluded point cloud in a deep-learning fashion. To the best of our knowledge, we are the first to utilize deep learning to tackle such a problem. In such a deep learning framework, double supervisions: points on the symmetry plane and normal vectors are employed to help us pinpoint the symmetry plane. We conducted experiments on the YCB-video dataset and demonstrate the efficacy of our method. 

## Requirements
Python 3.6, PyTorch 0.4.1

## Running
`sh experiments/scripts/train_ycb.sh`

## Symmetries
The symmetries of the standalone YCB objects are manually annotated in `symmetries.txt`, and adapted to decreasing order of the circumference of symmetry planes for better results in `symmetries_orders.txt`.

## Supplementary Repo
Our code is adapted from https://github.com/j96w/DenseFusion, in case of missing files and checkpoints, please refer to that repo.
