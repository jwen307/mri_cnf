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
from util import helper
from evals import metrics
import variables
import yaml


#Get the input arguments
args = helper.flags()

#Get the checkpoint arguments if needed
load_ckpt_dir = args.load_ckpt_dir
load_last_ckpt = args.load_last_ckpt


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
            rss=False

        report_path = os.path.join(load_ckpt_dir, 'metrics')
        Path(report_path).mkdir(parents=True, exist_ok=True)

        # Get the mean metrics
        mean_metrics = metrics.evaluate_mean(model, data, num_samples = 8, complex=False, temp=1.0, rss=rss)
        with open(os.path.join(report_path, 'metric_avg.yaml'), 'w') as file:
            documents = yaml.dump(mean_metrics, file)

        # Get the posterior metrics
        posterior_1 = metrics.evaluate_posterior(model, data.test_dataloader, num_samples = 8, temp=1.0, rss=rss)
        with open(os.path.join(report_path, 'metrics_posterior1.yaml'), 'w') as file:
            documents = yaml.dump(posterior_1, file)

        # Get the posterior 2 metrics
        posterior_2 = metrics.evaluate_posterior(model, data.val_dataloader, num_samples = 8, temp=0.5, rss=rss)
        with open(os.path.join(report_path, 'metrics_posterior2.yaml'), 'w') as file:
            documents = yaml.dump(posterior_2, file)


    except:

        traceback.print_exc()
       
        

