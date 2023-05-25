#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coupling.py
    - Modules to define components of the normalizing flow
"""
import torch
from typing import Callable

import FrEIA.modules as Fm


   
#This layer takes the conditional information and uses it to find the bias and scale
class AffineInjector(Fm.InvertibleModule):
    
    def __init__(self, dims_in, dims_c=[], subnet_constructor: Callable = None):
        
        super().__init__(dims_in, dims_c)
        
        self.channels, self.h, self.w = dims_in[0]
        
        assert all([tuple(dims_c[i][1:]) == tuple(dims_in[0][1:]) for i in range(len(dims_c))]), \
            "Dimensions of input and one or more conditions don't agree."
        self.conditional = (len(dims_c) > 0)
        self.condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])
        
        
        #Twice as many outputs because separate outputs for s and t
        self.subnet = subnet_constructor(self.condition_length,  self.channels*2)

        
    def forward(self, x, c=[], rev=False, jac = True):
        
        #x is passed as a list, so use x[0]
        x=x[0]
        

        #Pass the masked version in
        log_scale_shift = self.subnet(c[0])
        t = log_scale_shift[:,0::2,:,:]
        #s = torch.exp(log_scale_shift[:,1::2,:,:])
        s = torch.sigmoid_(log_scale_shift[:,1::2,:,:] + 2.0)

 
        #Apply the affine transformation
        if not rev:
            x = s*x + t
            log_det_jac = torch.log(s).sum(dim=[1,2,3])
            
        else:
            x = (x-t)/s
            log_det_jac = -torch.log(s).sum(dim=[1,2,3])
            
        return (x,), log_det_jac
        
        
    def output_dims(self, input_dims):
        '''See base class for docstring'''
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims    
