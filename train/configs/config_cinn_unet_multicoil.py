#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 15:13:37 2022

@author: jeff
Configuration file for training a conditional flow with a U-Net
"""

class Config():
    def __init__(self):

        self.config = {
            'data_args':
                {
                'mri_type': 'knee',  # brain or knee
                'center_frac': 0.08,
                'accel_rate': 4,
                'img_size': 320,
                'challenge': "multicoil",
                'complex': True, # if singlecoil, specify magnitude or complex
                'scan_type': 'CORPD_FBK',  # Knee: 'CORPD_FBK' Brain: 'AXT2'
                'mask_type': 'knee',  # Options :'s4', 'default', 'center_aug'
                'num_vcoils': 8,
                'acs_size': 13,  # 13 for knee, 32 for brain
                'slice_range': None,  # [0, 8], None
                },

            'train_args':
                {
                'lr': 5e-4,
                'batch_size': 4,
                'pretrain_unet': True
                },

            'flow_args':
                {
                'model_type': 'MulticoilCNF',
                'distribution': 'gaussian',
                'build_num': 1, #Our model is 1

                # Flow parameters
                'num_downsample': 3,
                'cond_conv_chs': [64, 64, 128],
                'downsample': 'squeeze',
                'num_blocks': 20,
                'use_fc_block': False,
                'num_fc_blocks': 2,
                'cond_fc_size': 64,
                'null_proj': True, #Use nullspace projection
                },

            'unet_args':
                {
                'chans': 128,
                'num_pool_layers': 4,
                'drop_prob': 0.0,
                'data_consistency': True,
                'unet_num_epochs': 50,
                'unet_lr': 3e-3,
                }
        }



