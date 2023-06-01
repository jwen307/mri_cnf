#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 06:56:12 2022

@author: jeff

train_cnf.py
    - Script to train a conditional normalizing flow
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers,seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os
from pathlib import Path
import traceback

import sys
sys.path.append("../")

from datasets.fastmri_multicoil import FastMRIDataModule
from util import helper, network_utils, viz
from evals import metrics
import variables
import yaml


#Get the input arguments
args = helper.flags()

#Get the checkpoint arguments if needed
load_ckpt_dir = args.load_ckpt_dir
load_last_ckpt = args.load_last_ckpt
# load_ckpt_dir = '/storage/jeff/mri_cnf/MulticoilCNF/version_2/'
# load_last_ckpt = False

# Determine which sample to use
samp_num = 10
num_samples= 8
temp = 1.0


if __name__ == "__main__":

    
    #Load the previous configurations
    ckpt_name = 'last.ckpt' if load_last_ckpt else 'best_bpd.ckpt'
    ckpt = os.path.join(load_ckpt_dir,
                        'checkpoints',
                        ckpt_name)

    #Get the configuration file
    config_file = os.path.join(load_ckpt_dir, 'config.pkl')
    config = helper.read_pickle(config_file)
    

    try:
        # Get the directory of the dataset
        base_dir = variables.fastmri_paths[config['data_args']['mri_type']]

        # Get the model type
        model_type = config['flow_args']['model_type']

        # Get the data
        data = FastMRIDataModule(base_dir,
                                 batch_size=config['train_args']['batch_size'],
                                 num_data_loader_workers=4,
                                 **config['data_args'],
                                 )
        data.prepare_data()
        data.setup()


        #Load the model
        model = helper.load_model(model_type, config, ckpt)

        # Use RSS for the knee dataset
        if config['data_args']['mri_type'] == 'knee':
            rss = True
        else:
            rss = False

        report_path = os.path.join(load_ckpt_dir, 'imgs')
        Path(report_path).mkdir(parents=True, exist_ok=True)

        # Get the data
        dataset = data.test
        # Get the data
        cond = dataset[samp_num][0].unsqueeze(0).to(model.device)
        gt = dataset[samp_num][1].unsqueeze(0).to(model.device)
        mask = dataset[samp_num][2].to(model.device)
        norm_val = dataset[samp_num][3].unsqueeze(0).to(model.device)
        maps = network_utils.get_maps(cond, model.acs_size, norm_val)

        # Generate the posteriors
        with torch.no_grad():
            samples = model.reconstruct(cond,
                                        num_samples,
                                        temp,
                                        check=True,
                                        maps=maps,
                                        mask=mask,
                                        norm_val=norm_val,
                                        split_num=4,
                                        multicoil=False,
                                        rss=rss)

        # Get the magnitude images
        gt_mag = network_utils.get_magnitude(network_utils.unnormalize(gt, norm_val), maps=maps, rss=rss).cpu()
        cond_mag = network_utils.get_magnitude(network_utils.unnormalize(cond, norm_val), maps=maps, rss=rss).cpu()

        # Get the mean of the posteriors
        mean = samples[0].mean(dim=0).unsqueeze(0).cpu()

        # Get the standard deviation map
        std = samples[0].std(dim=0).unsqueeze(0).cpu()

        # Save the images
        viz.save_posteriors(gt_mag, rotate=True, save_dir=os.path.join(report_path, 'gt'))
        viz.save_posteriors(cond_mag, rotate=True, save_dir=os.path.join(report_path, 'cond'))
        viz.save_posteriors(mean, rotate=True, save_dir=os.path.join(report_path, 'mean'))
        viz.save_posteriors(samples[0].cpu(), rotate=True, save_dir=os.path.join(report_path, 'samples'), num_samples=len(samples[0]))
        viz.save_posteriors(std, rotate=True, std=True, save_dir=os.path.join(report_path, 'std'))

        # Get the MAP estimate
        if config['flow_args']['model_type'] == 'MulticoilCNF':
            sample_MAP = model.reconstruct_MAPS(cond,
                                                maps=maps,
                                                mask=mask,
                                                norm_val=norm_val,
                                                rss=rss,
                                                epochs=100)

            # Save the image
            viz.save_posteriors(sample_MAP[0].cpu(), rotate=True, save_dir=os.path.join(report_path, 'MAP'))




    except:

        traceback.print_exc()
       
        

