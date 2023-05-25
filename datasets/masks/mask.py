#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mask.py
    - Helper function to load the masks and to apply the mask to the data
"""

import numpy as np
import scipy.io 
import torch
import sys
import os

sys.path.append('../../')
from util import helper

def get_mask(accel=4, size=320, mask_type='knee'):

    #Get the root of the project
    root = helper.get_root()

    #Mask location
    mask_loc = os.path.join(root, 'datasets/masks/mask_accel{0}_size{1}_gro_{2}.mat'.format(accel, size, mask_type))
    
    #Load the matrix
    #mask = scipy.io.loadmat('../datasets/masks/mask_accel{0}_size{1}_gro_{2}.mat'.format(accel, size, mask_type))
    mask = scipy.io.loadmat(mask_loc)
    mask = mask['samp'].astype('float32')
    mask = torch.from_numpy(mask).unsqueeze(0)
    
    
    return mask

def apply_mask(data, mask):
    
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data

