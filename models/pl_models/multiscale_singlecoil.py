#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:11:18 2022

@author: jeff
"""

import torch
import numpy as np
import pytorch_lightning as pl
import os
import time
import math
import fastmri
import sys

sys.path.append('../../')
from util import viz, network_utils, torch_losses


#Load modules for the UNet
from models.pl_models.base_cflow import _BaseCFlow
from models.pl_models.unet_multicoil import UNetMulticoil
from models.net_builds.build_unet import build0 as unetbuild
from models.networks.misc_nets import unet


class SinglecoilCNF(_BaseCFlow):

    def __init__(self, config):
        '''
        config: Configurations for the networks and training
        '''

        super().__init__(config)

        # Get the parameters
        self.acs_size = config['data_args']['acs_size']
        self.rss = config['flow_args']['rss']
        self.challenge = config['data_args']['challenge']
        self.complex = config['data_args']['complex']


    # rev = False is the normalizing direction, rev = True is generating direction
    def forward(self, x, c, rev=False, posterior_sample=False, mask=None, norm_val=None):
        '''

        :param x: Either a latent vector or a ground truth image
        :param c: Zero-fileld image
        :param rev: False is normalizing direction, True is generating direction
        :param posterior_sample: Generate posterior samples (conditional info is the same)
        :param mask: Sampling mask
        :param maps: Sensitivity maps
        :param norm_val: Normalizing value
        :return: z: Latent vector or prediction, ldj: Log-determinant of the Jacobian
        '''
        num_samples = c.shape[0]

        # If we're looking for posterior samples, just pass in one of the conditionals (they're repeats)
        if posterior_sample:
            c = c[0].unsqueeze(0)
            norm_val = norm_val[0].unsqueeze(0)

        # Switch to singlecoil and magnitude if needed
        if self.challenge == 'singlecoil':
            c = network_utils.unnormalize(c, norm_val)
            if not self.rss:
                maps = network_utils.get_maps(c, self.acs_size)
            else:
                maps = None
            c = network_utils.multicoil2single(c, maps=maps, rss=self.rss)


            # Switch to magnitude if needed
            if not self.complex:
                c = network_utils.get_magnitude(c)

            # Renormalize
            c = network_utils.normalize(c, norm_val)



        # Get the conditional information
        c = self.cond_net(c)

        if posterior_sample:
            # Repeat the conditional information at each layer for posterior samples
            for k in range(len(c)):
                c[k] = c[k].repeat(num_samples, 1, 1, 1)

        # Normalizing direction
        if not rev:
            # Make the input the correct input dimensions
            # Switch to singlecoil and magnitude if needed
            if self.challenge == 'singlecoil':
                x = network_utils.unnormalize(x, norm_val)
                x = network_utils.multicoil2single(x, maps=maps, rss=self.rss)

                # Switch to magnitude if needed
                if not self.complex:
                    x = network_utils.get_magnitude(x)

                # Renormalize
                x = network_utils.normalize(x, norm_val)

            z, ldj = self.flow(x, c, rev=rev)

        # Generating direction
        else:
            z, ldj = self.flow(x, c, rev=rev)

        return z, ldj

    def reconstruct(self, c, num_samples, temp=1.0, check=False, maps=None, mask=None, norm_val=None,
                    split_num=None, rss=False, multicoil=False):
        '''

        :param c: conditional images (zero-filled)
        :param num_samples: Number of posterior samples
        :param temp: Scale factor for the distribution
        :param check: Check for NaN reconstructions?
        :param maps: Sensitivity maps
        :param mask: Sampling mask
        :param norm_val: Normalizing value
        :param split_num: Number to split the posterior sample generation by to fit in memory
        :param rss: Return the root-sum-of-squares reconstruction
        :param multicoil: Return multicoil reconstructions
        :return: (list) recon: List of reconstructions
        '''

        # Get the batch_size
        b = c.shape[0]

        #Collect the reconstructions
        recons = []

        # Figure out how to split the samples so it fits in memory
        if split_num is not None and num_samples > split_num:
            num_splits = math.ceil(num_samples / split_num)
        else:
            num_splits = 1

        total_samples = num_samples * 1.0

        # Process each conditional image separately
        for i in range(b):
            with torch.no_grad():

                all_splits = []

                # Generate posteriors in batches so they fit in memory
                for k in range(num_splits):
                    if split_num is not None:
                        # Check if there's less than 16 samples left
                        if (total_samples - k * split_num) < split_num:
                            num_samples = int(total_samples - k * split_num)
                        else:
                            num_samples = split_num

                    # Draw samples from the distribution
                    z = self.sample_distrib(num_samples, temp=temp)

                    # Repeat the info for each sample
                    c_in = c[i].unsqueeze(0).repeat(num_samples, 1, 1, 1)
                    norm_val_rep = norm_val[i].unsqueeze(0).repeat(num_samples, 1, 1, 1)

                    # Get the reconstructions
                    recon, _, = self.forward(z, c_in, rev=True,
                                                posterior_sample=True,
                                                mask=mask, norm_val=norm_val_rep,
                                                )

                    # Check if the reconstructions are valid
                    if check:
                        # Replace the invalid reconstructions
                        recon = self.check_recon(recon, c[i], temp, mask=mask, norm_val=norm_val[i])

                    # Unccomment to apply data consistency (only works if complex)
                    # Apply data consistency
                    # recon, _ = network_utils.get_dc_image(recon, c_in,
                    #                                       mask=mask,
                    #                                       norm_val=norm_val_rep)

                    # Unnormalize
                    recon = network_utils.unnormalize(recon ,norm_val[i])

                    if self.challenge == 'multicoil':
                        # Combine the coils
                        if not multicoil:
                            recon = network_utils.format_multicoil(recon, chans=False)
                            if rss:
                                recon = fastmri.rss_complex(recon, dim=1).unsqueeze(1)
                            else:
                                rep_maps = network_utils.check_type(maps[i], 'tensor').unsqueeze(0).repeat(num_samples, 1, 1, 1)
                                recon = network_utils.multicoil2single(recon, rep_maps)
                                recon = network_utils.get_magnitude(recon)

                    elif self.complex:
                        if not multicoil: # If you want the complex output
                            # Get the magnitude
                            recon = network_utils.get_magnitude(recon)

                    # Collect all the splits
                    all_splits.append(recon)

                recons.append(torch.cat(all_splits, dim=0))

        # Output is a list where each element is a tensor with the reconstructions for a single conditional image
        return recons


    # Make sure the reconstructions don't have NaN
    def check_recon(self, recon, cond, temp, mask=None, norm_val=None):

        for i in range(recon.shape[0]):
            num_nan = 0

            # Check if the reconstruction is valid
            while recon[i].abs().max() > 10 or torch.isnan(recon[i].abs().max()):
                with torch.no_grad():
                    # Draw samples from a distribution
                    z = self.sample_distrib(1, temp=temp)
                    recon[i], _ = self.forward(z, cond.unsqueeze(0), rev=True,
                                               mask=mask,
                                               norm_val=norm_val.unsqueeze(0))

                    num_nan += 1

                    if num_nan > 10:
                        if i == 0:
                            print('Cant find inverse for img')
                            #viz.show_img(cond.unsqueeze(0))
                            #viz.show_img(recon[i])
                        break

        return recon


    def configure_optimizers(self):

        cnf_opt = torch.optim.Adam(
            list(self.flow.parameters()) + list(self.cond_net.parameters()),
            lr=self.config['train_args']['lr'],
            weight_decay=1e-7
            )


        return cnf_opt


    def training_step(self, batch, batch_idx):

        # Get all the inputs
        c = batch[0].to(self.device)
        x = batch[1].to(self.device)
        masks = batch[2].to(self.device)
        norm_val = batch[3].to(self.device)


        # Pass through the CNF
        z, ldj = self(x, c, rev=False, norm_val=norm_val)

        # Find the negative log likelihood
        loss = self.get_nll(z, ldj, give_bpd=True)

        # Log the training loss
        self.log('train_loss', loss, prog_bar=True)


        return loss

    def validation_step(self, batch, batch_idx):

        c = batch[0]
        x = batch[1]
        masks = batch[2]
        norm_val = batch[3]

        board = self.logger.experiment

        with torch.no_grad():
            z, ldj = self(x, c, rev=False, norm_val=norm_val)

            # Find the negative log likelihood
            loss = self.get_nll(z, ldj, give_bpd=True)
            self.log('val_bpd', loss, sync_dist=True)

            # Show some example images
            if batch_idx == 1:
                # TODO: Note: the sensitivity map estimation messes with multi-gpu checkpointign so do RSS here
                # Show the original images and reconstructed images
                gt_grid = viz.show_img(x.float().detach().cpu(), return_grid=True, rss=True)
                board.add_image('GT Images', gt_grid, self.current_epoch)

                recons = self.reconstruct(c, num_samples=2, mask=masks, norm_val=norm_val, rss=True)

                # Concatenate the list
                recons = torch.cat(recons, dim=0)

                # Show the images
                grid = viz.show_img(recons.float().detach().cpu(), return_grid=True)
                board.add_image('Full Val Image', grid, self.current_epoch)



# Uncomment to test model
# from train.configs.config_singlecoil import Config
# from datasets.fastmri_multicoil import FastMRIDataModule
# #%% Test out the model to make sure it runs
# if __name__ == '__main__':
#     # Get the configurations
#     conf = Config()
#     config = conf.config
#
#     if config['data_args']['mri_type'] == 'brain':
#         base_dir = "/storage/fastMRI_brain/data/"
#     elif config['data_args']['mri_type'] == 'knee':
#         base_dir = "/storage/fastMRI/data/"
#     else:
#         raise Exception("Please specify an mri_type in config")
#
#     # Get the data
#     data = FastMRIDataModule(base_dir,
#                              num_data_loader_workers=4,
#                              **config['data_args'],
#                              )
#     data.prepare_data()
#     data.setup()
#
#
#     # Initialize the network
#     model = SinglecoilCNF(config)
#
#     dataset = data.val
#     model = model.cuda()
#     model = model.eval()
#     samp_num=0
#
#     # Get the data
#     cond = dataset[samp_num][0].unsqueeze(0).to(model.device)
#     gt = dataset[samp_num][1].unsqueeze(0).to(model.device)
#     mask = dataset[samp_num][2].to(model.device)
#     norm_val = torch.tensor(dataset[samp_num][3]).unsqueeze(0).to(model.device)
#     maps = network_utils.get_maps(cond, model.acs_size, norm_val)
#
#     with torch.no_grad():
#         z, ldj = model(gt, cond, rev=False, mask=mask, norm_val=norm_val[0])