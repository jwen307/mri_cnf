#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 06:56:12 2022

@author: jeff
"""
# %%

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
import os
import yaml
from pathlib import Path
import traceback
import numpy as np
import fastmri
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys

sys.path.append("..")



# TODO Change these for new models
from models.pl_models.multiscale_unet_multicoil import MultiscaleUnetMulticoil
from models.net_builds import build_multiscale_nets
from train.configs.config_cinn_unet_multicoil import Config

from datasets.fastmri_multicoil import FastMRIDataModule
from util import helper, viz, network_utils, mail
import argparse
from evals.metrics import evaluate, evaluate_temps
from evals import fid, metrics

import evals.metrics

# Location of the dataset
base_dir = "/storage/fastMRI_brain/data/"
# base_dir= '/scratch/fastMRI/data/'
# base_dir = '/storage/data/'
# base_dir = '../../datasets/fastMRI/data'

'''
#Get the input arguments
args = util.bijflags()

train_net = args.train_net
load_ckpt_dir = args.load_ckpt_dir
load_ckpt_epoch = args.ckpt_epoch
load_ckpt_step = args.ckpt_step
'''
train_net = False
# load_ckpt_dir = 'None'
# load_ckpt_dir = '../logs/lightning_logs/version_28' #'../logs/default/version_3' or 'None' /storage/jeff/mri_cinn1/logs/
ckpt_dir = '/storage/jeff/mri_cinn1/logs/lightning_logs/'
load_version = 'version_{}'.format(76)
# load_ckpt_dir = '/storage/jeff/mri_cinn1/logs/lightning_logs/version_60'
# load_ckpt_dir='None'
load_ckpt_epoch = 49
load_ckpt_step = 76400
load_ckpt_dir = os.path.join(ckpt_dir, load_version)
eval_dir = os.path.join('../logs/lightning_logs', load_version)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

torch.cuda.is_available()

# def get_examples(model, model_path, dataset):
#
#     model.freq_sep = True
#
#     #Specify the index of the images to test
#     img_idxs = [25,100]
#
#     model.cuda()
#     model.eval()
#
#     for img_idx in img_idxs:
#
#         #Get the data
#         cond = dataset[img_idx][0].unsqueeze(0)
#         gt = dataset[img_idx][1].unsqueeze(0)
#         #maps = np.expand_dims(dataset[img_idx][2], axis=0)
#         maps = network_utils.get_maps(cond, model.acs_size)
#         masks = dataset[img_idx][3]
#         std = torch.tensor(dataset[img_idx][4]).unsqueeze(0)
#
#         with torch.no_grad():
#             #model.freq_sep = True
#             #Get a few posterior samples
#             samples, fsrecon = model.reconstruct(cond.to(model.device),
#                                         num_samples = 50,
#                                         maps = maps,
#                                         mask = masks.to(model.device),
#                                         std = std.to(model.device),
#                                         temp=1.0,
#                                         give_fs = True,
#                                         split_samples=True
#                                         )
#             '''
#             if model.freq_sep:
#                 if torch.is_tensor(maps):
#                     rep_maps = maps[0].unsqueeze(0).repeat(50,1,1,1)
#                 else:
#                     rep_maps = np.repeat(np.expand_dims(maps[0],axis=0), 10, axis=0)
#
#                 z = model.sample_distrib(num_samples = 50, temp = 1)
#                 outputs, _ = model(z, cond.repeat(50,1,1,1).to(model.device), rev=True,
#                                    maps = rep_maps,
#                                    posterior_sample = False,
#                                    mask= masks.to(model.device),
#                                    std= std[0].unsqueeze(0).repeat(10,1,1,1).to(model.device))
#             '''
#
#         #Save all the samples
#         img_path = os.path.join(model_path, 'imgs')
#         if not os.path.exists(img_path):
#             os.makedirs(img_path)
#
#         val_range = None #(0,2.0)
#
#         # Get the single coil magnitude ground truth
#         gt = gt * std[0]
#         gt_single = network_utils.multicoil2single(gt, maps)
#         gt_mag = fastmri.complex_abs(gt_single)
#
#         sample_mag = fastmri.complex_abs(samples[0][0].permute(1,2,0)).unsqueeze(0).cpu()
#         mean_mag = fastmri.complex_abs(samples[0].mean(dim=0).permute(1,2,0)).unsqueeze(0).cpu()
#
#         sample_psnr = recon_metrics1.psnr(gt_mag.numpy(), sample_mag.numpy())
#         mean_psnr = recon_metrics1.psnr(gt_mag.numpy(), mean_mag.numpy())
#
#         print(sample_psnr)
#         print(mean_psnr)
#
#
#         viz.show_img(samples[0],  val_range=val_range,
#                      save_dir = os.path.join(img_path, 'posterior_samples{0}.png'.format(img_idx)))
#
#
#         viz.show_multicoil_combo(gt, maps, val_range=val_range,
#                      save_dir = os.path.join(img_path, 'gt{0}.png'.format(img_idx)))
#
#
#
#         viz.show_multicoil_combo(cond[0], maps, val_range=val_range,
#                      save_dir = os.path.join(img_path, 'zerofilled{0}.png'.format(img_idx)))
#
#
#
#         viz.show_img(samples[0][0], val_range=val_range,
#                      title=f'PSNR: {sample_psnr}',
#                      save_dir = os.path.join(img_path, 'posterior_sample{0}.png'.format(img_idx)))
#
#
#
#         viz.show_img(samples[0].mean(dim=0), val_range=val_range,
#                      title=f'PSNR: {mean_psnr}',
#                      save_dir = os.path.join(img_path, 'sample_mean{0}.png'.format(img_idx)))
#
#         '''
#         if model.freq_sep:
#             viz.show_multicoil_combo(outputs.mean(dim=0), maps, val_range = (0,1),
#                      save_dir = os.path.join(img_path, 'output_mean{0}.png'.format(img_idx)))
#         '''
#
#         #gt_single = network_utils.multicoil2single(gt, maps).permute(0,3,1,2)
#         #gt_masked= network_utils.apply_mask(gt_single, masks, std=std, inv_mask = True)
#         #viz.show_img(gt_masked)
#         gt_masked= network_utils.apply_mask(gt, masks, std=std, inv_mask = True)
#         viz.show_multicoil_combo(gt_masked, maps)
#         gt_masked_single = network_utils.multicoil2single(gt_masked, maps)
#         gt_masked_single_mag = fastmri.complex_abs(gt_masked_single)
#         recon_fs = network_utils.multicoil2single(fsrecon[0], maps)
#         recon_fs_mag = fastmri.complex_abs(recon_fs)
#         viz.show_multicoil_combo(fsrecon[0], maps)
#         viz.show_error_map(recon_fs_mag.cpu(),gt_masked_single_mag.cpu(), title='FS Error Map for cINN Posterior', limits=(0,0.5))
#
#         cond_masked = network_utils.apply_mask(cond, masks, std=std, inv_mask = True)
#         viz.show_multicoil_combo(cond_masked, maps)
#         cond_masked_single = network_utils.multicoil2single(cond_masked, maps)
#         cond_masked_single_mag = fastmri.complex_abs(cond_masked_single)
#         viz.show_error_map(cond_masked_single_mag.cpu(),gt_masked_single_mag.cpu(), title='FS Error Map for UNet', limits=(0,0.5))
#
#
#
#         unnorm_samples = samples[0] #* std[0]
#         capsd = unnorm_samples.std(dim=0).mean()
#         #if not model.freq_sep:
#         unnorm_samples = unnorm_samples.permute(0,2,3,1)
#
#         unnorm_mag_samples = fastmri.complex_abs(unnorm_samples).unsqueeze(1)
#         apsd = unnorm_mag_samples.std(dim=0).mean()
#         viz.show_std_map(samples[0].cpu(), title='Standard Dev Map, cAPSD= {0:.3e}, APSD={1:.3e}'.format(capsd, apsd))
#
#         #Look at the error between the mean and gt
#         #gt = gt * std[0]
#         #gt_single = network_utils.multicoil2single(gt, maps)
#         #gt_mag = fastmri.complex_abs(gt_single)
#         #if not model.freq_sep:
#         mean_mag = fastmri.complex_abs(samples[0].mean(dim=0).permute(1,2,0))
#         #else:
#         #mean_mag = fastmri.complex_abs(samples[0].mean(dim=0))
#
#         viz.show_error_map(mean_mag.cpu().unsqueeze(0),gt_mag.cpu(), title='Error Map')#, limits=(0,0.5))
#
#         plt.hist(gt_mag.flatten() - mean_mag.cpu().flatten(), bins=1000)
#         plt.title('Error Histogram: {0:.3e}'.format(torch.sum(torch.abs(mean_mag.cpu().unsqueeze(0)-gt_mag.cpu()))))
#         plt.legend()
#         plt.show()


# %% Get the arguments for training
# if __name__ == "__main__":

# Use new configurations if not loading a pretrained model
if load_ckpt_dir == 'None':
    # Get the configurations
    config = Config()
    net_args = config.net_args
    ckpt = None

# Load the previous configurations
else:
    # Get the path to the checkpoint
    # ckpt = torch.load(os.path.join(load_ckpt_dir,
    #                    'checkpoints', 
    #                    'epoch={0}.ckpt'.format(load_ckpt_epoch)))

    ckpt = os.path.join(load_ckpt_dir,
                        'checkpoints',
                        'epoch={0}-step={1}.ckpt'.format(load_ckpt_epoch, load_ckpt_step))

    # Get the configuration file
    config_file = os.path.join(load_ckpt_dir, 'config.pkl')
    net_args = util.read_pickle(config_file)

    # net_args['train_args']['lr'] = 2e-4
    net_args['train_args']['mri_type'] = 'brain'
    # net_args['train_args']['mask_type'] = 'matt2'

try:

    # Get the data
    data = FastMRIDataModule(base_dir,
                             num_data_loader_workers=int(os.cpu_count() / 4),
                             **net_args['train_args'],
                             )
    data.prepare_data()
    data.setup()

    # Setup the network
    if 'num_vcoils' in net_args['train_args']:
        in_ch = net_args['train_args']['num_vcoils'] * 2
    else:
        in_ch = 16

    # Set the input dimensions
    img_size = net_args['train_args']['img_size']
    input_dims = [in_ch, img_size, img_size]

    # Pick which build functions to use
    builds = [
        build_multiscale_nets.buildbij0,
        build_multiscale_nets.buildbij1,
        build_multiscale_nets.buildbij2,
        build_multiscale_nets.buildbij3,
        build_multiscale_nets.buildbij4,
        build_multiscale_nets.buildbij5,
        build_multiscale_nets.buildbij6,
    ]
    build_num = net_args['train_args']['build_num']

    # Monitor the learning rate
    seed_everything(1, workers=True)
    lr_monitor = LearningRateMonitor(logging_interval=None)

    # Load a checkpoint if specified
    if load_ckpt_dir != 'None':
        print("Loading previous checkpoint")
        # model.load_state_dict(ckpt['state_dict'])
        model = MultiscaleUnetMulticoil.load_from_checkpoint(ckpt, strict=False, input_dims=input_dims,
                                                             build_bij_func=builds[build_num],
                                                             net_args=net_args)

    else:
        # Initialize the network
        model = MultiscaleUnetMulticoil(input_dims,
                                        builds[build_num],
                                        net_args
                                        )

    # Print the number of parameters
    print("Flow")
    network_utils.print_num_params(model.flow)
    print("Bijective Conditional")
    network_utils.print_num_params(model.bij_cnet)

    # Create the tensorboard logger
    # logger = loggers.TensorBoardLogger("../logs/")
    logger = loggers.TensorBoardLogger("/storage/jeff/mri_cinn1/logs/")

    if train_net:

        ckpt_callback = ModelCheckpoint(
            save_top_k=3,
            monitor='val_bpd',
            mode='min',
        )

        # Create the trainers
        trainer = pl.Trainer(
            max_epochs=1,
            gradient_clip_val=1.0,
            accelerator='gpu',
            logger=logger,
            check_val_every_n_epoch=1,
            callbacks=[lr_monitor, ckpt_callback, DeviceStatsMonitor()],
            strategy='ddp',
            detect_anomaly=True,
            # precision=64,
            # overfit_batches=0.01
            # detect_anomaly=True,
        )

        # Save the configurations
        model_path = trainer.logger.log_dir
        Path(model_path).mkdir(parents=True, exist_ok=True)
        config_file = os.path.join(model_path, 'config.pkl')
        util.write_pickle(net_args, config_file)

        if ckpt is None:
            print("Starting Training")
            trainer.fit(model, data.train_dataloader(), data.val_dataloader())

        else:
            print("Resuming Training")
            trainer.fit(model, data.train_dataloader(), data.val_dataloader(), ckpt_path=ckpt)

    else:
        # model_path = load_ckpt_dir
        model_path = eval_dir
    # Send an alert that the model is done training
    # mail.send_mail("Training Complete", "The training is complete")
    # get_examples(model, model_path, data.val)
    # model.freq_sep = True
    # evaluate(model, model_path, data, num_samples=50, model_type='cinn', temp = 1.0)
    # evaluate_temps(model, model_path, data, num_samples=[1,2,4,8,16,32], model_type='cinn', temps = [1])



except:
    # mail.send_mail("Training Error", 'Training has stopped')
    traceback.print_exc()

# %% Try looking for the MAP estimate
dataset = data.test
img_idx = 0
epochs = 500


model.cuda()
model.eval()

# Get the data
cond = dataset[img_idx][0].unsqueeze(0).to(model.device)
gt = dataset[img_idx][1].unsqueeze(0).to(model.device)
# maps = np.expand_dims(dataset[img_idx][2], axis=0)
maps = network_utils.get_maps(cond, model.acs_size)
masks = dataset[img_idx][3].to(model.device)
std = torch.tensor(dataset[img_idx][4]).unsqueeze(0)

#recon = model.reconstruct_MAPS(cond, maps, masks, std, epochs=500)

# with torch.no_grad():
#     #model.freq_sep = True
#     #Get a few posterior samples
#     recon, fsrecon = model.reconstruct(cond.to(model.device),
#                                 num_samples = 1,
#                                 maps = maps,
#                                 mask = masks.to(model.device),
#                                 std = std.to(model.device),
#                                 temp=1.0,
#                                 give_fs = True,
#                                 split_samples=True
#                                 )

# Start at the mean image
z = model.sample_distrib(1, temp=1.0).to(model.device)
with torch.no_grad():
    x, ldj = model(z, cond, rev=True, mask=masks, maps=maps, std=std[0])

# Define the optimization
x = x.requires_grad_(True)
opt = torch.optim.Adam([x], lr=0.1)

# Learning rate scheduler
sched_factor = 0.9
sched_patience = 10
sched_thresh = 0.001
sched_cooldown = 1
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, factor=sched_factor,
    patience=sched_patience, threshold=sched_thresh,
    min_lr=1e-9, eps=1e-08, cooldown=sched_cooldown,
    verbose=False)



for i in range(epochs):
    # Zero the gradient
    opt.zero_grad()

    # Find the latent vector
    z, ldj = model(x, cond, rev=False, mask=masks, maps=maps, std=std[0])

    # Get the loss
    loss = model.get_nll(z, ldj, give_bpd=True)

    #For projected MAP
    #loss = (-ldj / (np.prod(z.shape[1:]) * np.log(2)) + 8) + torch.abs(z.norm() - 1536)
    print('Epoch: {0}, Loss: {1}, Norm: {2}'.format(i, loss, z.detach().norm()))

    # Backpropagate and update
    loss.backward()
    opt.step()

    # Update the learning rate
    sched.step(loss)

    #Project onto the sphere
    # with torch.no_grad():
    #     z, ldj = model(x, cond, rev=False, mask=masks, maps=maps, std=std[0])
    #     z = z / z.norm() * torch.sqrt(torch.tensor(z.shape[1]))
    #     x, ldj = model(z, cond, rev=True, mask=masks, maps=maps, std=std[0])

# Get the image
x = x.detach().cpu()
recon, _ = network_utils.get_dc_image(x, cond.cpu(), masks.cpu(), std=std)
recon = recon * std
recon = network_utils.multicoil2single(recon, maps)
#recon = recon[0].permute(0,2,3,1).cpu()
viz.show_img(recon)

# Get the single coil magnitude ground truth
gt = gt * std[0]
gt_single = network_utils.multicoil2single(gt, maps)
gt_mag = fastmri.complex_abs(gt_single)

# Get the single coil magnitude condition
cond = cond * std[0]
cond_single = network_utils.multicoil2single(cond, maps)
cond_mag = fastmri.complex_abs(cond_single)

# Get the magnitude image of the mean
mean_mag = fastmri.complex_abs(recon)

# Save all the samples
#img_path = os.path.join(model_path, 'imgs/{}_post'.format(img_idx))
img_path = os.path.join(model_path, 'imgs/{}_map'.format(img_idx))
if not os.path.exists(img_path):
    os.makedirs(img_path)

viz.save_all_imgs(gt_mag, cond_mag, mean_mag, img_path, rotate=True)


# %% Try looking for the MAP estimate norms
dataset = data.test
epochs = 1000

model.cuda()
model.eval()

all_norms = []

for img_idx in tqdm(range(0,10)):
    # Get the data
    cond = dataset[img_idx][0].unsqueeze(0).to(model.device)
    gt = dataset[img_idx][1].unsqueeze(0).to(model.device)
    # maps = np.expand_dims(dataset[img_idx][2], axis=0)
    maps = network_utils.get_maps(cond, model.acs_size)
    masks = dataset[img_idx][3].to(model.device)
    std = torch.tensor(dataset[img_idx][4]).unsqueeze(0)

    recon, norms = model.reconstruct_MAPS(cond, maps, masks, std, epochs=epochs, give_norm=True)
    all_norms.append(norms[0].detach().cpu())

norms_all = torch.stack(all_norms)
plt.hist(norms_all)
plt.show()

# %% Save images
# img_idxs = [i for i in range(18,19)]
# rotate = True
# model.freq_sep = True
# dataset = data.test
#
# model.cuda()
# model.eval()
#
# for img_idx in img_idxs:
#
#     #Get the data
#     cond = dataset[img_idx][0].unsqueeze(0)
#     gt = dataset[img_idx][1].unsqueeze(0)
#     #maps = np.expand_dims(dataset[img_idx][2], axis=0)
#     maps = network_utils.get_maps(cond, model.acs_size)
#     masks = dataset[img_idx][3]
#     std = torch.tensor(dataset[img_idx][4]).unsqueeze(0)
#
#     with torch.no_grad():
#         #model.freq_sep = True
#         #Get a few posterior samples
#         samples, fsrecon = model.reconstruct(cond.to(model.device),
#                                     num_samples = 32,
#                                     maps = maps,
#                                     mask = masks.to(model.device),
#                                     std = std.to(model.device),
#                                     temp=1.0,
#                                     give_fs = True,
#                                     split_samples=True
#                                     )
#
#     #Save all the samples
#     img_path = os.path.join(model_path, 'imgs/{}'.format(img_idx))
#     if not os.path.exists(img_path):
#         os.makedirs(img_path)
#
#     # Get the single coil magnitude ground truth
#     gt = gt * std[0]
#     gt_single = network_utils.multicoil2single(gt, maps)
#     gt_mag = fastmri.complex_abs(gt_single)
#
#     # Get the single coil magnitude condition
#     cond = cond * std[0]
#     cond_single = network_utils.multicoil2single(cond, maps)
#     cond_mag = fastmri.complex_abs(cond_single)
#
#     #Get the magnitude image of the mean
#     mean_mag = fastmri.complex_abs(samples[0].mean(dim=0).permute(1,2,0)).unsqueeze(0).cpu()
#
#     viz.save_all_imgs(gt_mag, cond_mag, mean_mag, img_path, rotate=rotate)
#
#     #Get the standard deviation map
#     samples_mag = fastmri.complex_abs(samples[0].permute(0,2,3,1)).cpu()
#     samples_std = samples_mag.std(dim=0).unsqueeze(0)
#
#     viz.save_std_map(samples_std, colorbar=False, save_dir = img_path, rotate=rotate, limits=[0,3e-5])
#
#     #Save the posterior samples with zoomed in regions
#     rect_colors = ['deepskyblue', 'y']
#     rect = [[175, 160, 30, 30], [172,200, 30, 30]] #[x-top right corner, y-top right corner, x-width, y-width] where x is right left, y is top down
#     viz.save_posteriors(samples_mag, rotate=rotate, save_dir = os.path.join(img_path, 'posteriors'),
#                         rect = rect, rect_colors = rect_colors, num_samples=20)
#     viz.save_posteriors(gt_mag, rotate=rotate, save_dir=os.path.join(img_path, 'gt'),
#                         rect=rect, rect_colors=rect_colors)
#     viz.save_posteriors(cond_mag, rotate=rotate, save_dir=os.path.join(img_path, 'cond'),
#                         rect=rect, rect_colors=rect_colors)
#     viz.save_posteriors(samples_std, rotate=rotate, std= True, save_dir=os.path.join(img_path, 'std'),
#                         val_range=(0,3e-5),
#                         rect=rect, rect_colors=rect_colors)

# %% Find the PSNR and SSIM for 72 random test images
# from tqdm import tqdm
# import time
#
# samples = torch.randint(len(data.test), (1,72))
# idxs = torch.randperm(len(data.val))[0:72].numpy()
#
# model.cuda()
# model.eval()
#
# dataset = data.test
#
# sample_psnrs = []
# mean_psnrs = []
# sample_ssims = []
# mean_ssims = []
# times = []
#
# #for k, samp_num in tqdm(enumerate(idxs)):
# for samp_num in tqdm(range(0,len(dataset))):
#     # Get the data
#     cond = dataset[samp_num][0].unsqueeze(0)
#     gt = dataset[samp_num][1].unsqueeze(0)
#     # maps = np.expand_dims(dataset[img_idx][2], axis=0)
#     maps = network_utils.get_maps(cond, model.acs_size)
#     masks = dataset[samp_num][3]
#     std = torch.tensor(dataset[samp_num][4]).unsqueeze(0)
#
#     with torch.no_grad():
#         # model.freq_sep = True
#         start = time.time()
#         # Get a few posterior samples
#         samples, fsrecon = model.reconstruct(cond.to(model.device),
#                                              num_samples=32,
#                                              maps=maps,
#                                              mask=masks.to(model.device),
#                                              std=std.to(model.device),
#                                              temp=1.0,
#                                              give_fs=True,
#                                              split_samples=True,
#                                              rss=False,
#                                              #multicoil=True
#                                              )
#
#         end = time.time()
#         times.append(end-start)
#
#         # Get the single coil magnitude ground truth
#         gt = gt * std[0]
#         gt_single = network_utils.multicoil2single(gt, maps)
#         gt_mag = fastmri.complex_abs(gt_single)
#
#         sample_mag = fastmri.complex_abs(samples[0][0].permute(1, 2, 0)).unsqueeze(0).cpu()
#         mean_mag = fastmri.complex_abs(samples[0].mean(dim=0).permute(1, 2, 0)).unsqueeze(0).cpu()
#         #mean_mag = samples[0].mean(dim=0).permute(1, 2, 0).unsqueeze(0).cpu()
#         # sample_single = network_utils.multicoil2single(samples[0][0].unsqueeze(0), maps)
#         # sample_mag = fastmri.complex_abs(sample_single)
#         # mean_sample = network_utils.multicoil2single(samples[0].mean(dim=0).unsqueeze(0),maps)
#         # mean_mag = fastmri.complex_abs(mean_sample)
#
#
#         sample_psnr = recon_metrics1.psnr(gt_mag.numpy(), sample_mag.numpy())
#         mean_psnr = recon_metrics1.psnr(gt_mag.numpy(), mean_mag.numpy())
#         #mean_psnr = recon_metrics1.psnr_complex(gt_single.numpy(), mean_mag.numpy())
#
#         sample_ssim = recon_metrics1.ssim(gt_mag.numpy(), sample_mag.numpy())
#         mean_ssim = recon_metrics1.ssim(gt_mag.numpy(), mean_mag.numpy())
#         #mean_ssim = recon_metrics1.ssim(gt_single.numpy(), mean_mag.numpy())
#
#         sample_psnrs.append(sample_psnr)
#         mean_psnrs.append(mean_psnr)
#         sample_ssims.append(sample_ssim)
#         mean_ssims.append(mean_ssim)
#
#
#
# s_psnr_mean = np.array(sample_psnrs).mean()
# m_psnr_mean = np.array(mean_psnrs).mean()
# s_ssim_mean = np.array(sample_ssims).mean()
# m_ssim_mean = np.array(mean_ssims).mean()
#
# s_psnr_stderr = float(np.std(sample_psnrs) / np.sqrt(len(sample_psnrs)))
# m_psnr_stderr = float(np.std(mean_psnrs) / np.sqrt(len(mean_psnrs)))
# s_ssim_stderr = float(np.std(sample_ssims) / np.sqrt(len(sample_ssims)))
# m_ssim_stderr = float(np.std(mean_ssims) / np.sqrt(len(mean_ssims)))
#
# print('Single Sample PSNR: {0} +/- {1}'.format(s_psnr_mean, s_psnr_stderr))
# print('Mean Sample PSNR: {0} +/- {1}'.format(m_psnr_mean, m_psnr_stderr))
# print('Single Sample SSIM: {0} +/- {1}'.format(s_ssim_mean, s_ssim_stderr))
# print('Mean Sample SSIM: {0} +/- {1}'.format(m_ssim_mean, m_ssim_stderr))


# %%

# rep_maps = np.repeat(np.expand_dims(maps,axis=0), 8, axis=0)
# with torch.no_grad():
#     z, ldj = model(fsrecon[0][0:8].cuda(),cond[0].unsqueeze(0).repeat(8,1,1,1).cuda(), rev=False, mask=masks.cuda(), posterior_sample=True, maps=rep_maps,std= std[0].unsqueeze(0).repeat(8,1,1,1))
#
# log_pz = -0.5*torch.sum(z**2, 1) - (0.5 * np.prod(z.shape[1:]) * torch.log(torch.tensor(2*torch.pi)))
# log_px = log_pz + ldj

# %% Look at PSNR vs Number of Samples
# from tqdm import tqdm
# def calc_metrics(reconstructions, targets, is_complex=False):
#     nmses = []
#     psnrs = []
#     ssims = []
#     rsnrs = []
#     for i in tqdm(range(len(reconstructions))):
#         nmses.append(recon_metrics1.nmse(targets[i], reconstructions[i]))
#         if is_complex:
#             psnrs.append(recon_metrics1.psnr_complex(targets[i], reconstructions[i]))
#         else:
#             psnrs.append(recon_metrics1.psnr(targets[i], reconstructions[i]))
#
#         ssims.append(recon_metrics1.ssim(targets[i], reconstructions[i]))
#         rsnrs.append(recon_metrics1.rsnr(targets[i], reconstructions[i]))
#
#     report_dict = {
#         'results': {
#             'mean_nmse': float(np.mean(nmses)),
#             'std_err_nmse': float(np.std(nmses) / np.sqrt(len(nmses))),
#             'mean_psnr': float(np.mean(psnrs)),
#             'std_err_psnr': float(np.std(psnrs) / np.sqrt(len(psnrs))),
#             'mean_ssim': float(np.mean(ssims)),
#             'std_err_ssim': float(np.std(ssims) / np.sqrt(len(ssims))),
#             'mean_rsnr (db)': float(np.mean(rsnrs)),
#             'std_err_rsnr (db)': float(np.std(rsnrs) / np.sqrt(len(rsnrs))),
#         }
#     }
#
#     return report_dict
#
# dataset = data.test
# model = model.cuda()
# model = model.eval()
#
# complex = False
#
# gts = []
# preds = [[] for i in range(6)]
#
# for samp_num in tqdm(range(0,len(dataset))):
#     # Get the data
#     cond = dataset[samp_num][0].unsqueeze(0)
#     gt = dataset[samp_num][1].unsqueeze(0)
#     #maps = np.expand_dims(dataset[samp_num][2], axis=0)
#
#     mask = dataset[samp_num][3]
#     std = torch.tensor(dataset[samp_num][4]).unsqueeze(0)
#     maps = network_utils.get_maps(cond, model.acs_size, std)
#
#     with torch.no_grad():
#         # model.freq_sep = True
#         # Get a few posterior samples
#         samples, fsrecon = model.reconstruct(cond.to(model.device),
#                                              num_samples=32,
#                                              maps=maps,
#                                              mask=mask.to(model.device),
#                                              std=std.to(model.device),
#                                              temp=1.0,
#                                              give_fs=True,
#                                              split_samples=True,
#                                              rss=False,
#                                              )
#
#     # Get the single coil magnitude ground truth
#     # Get the single coil magnitude ground truth
#     gt = gt * std[0]
#     gt_single = network_utils.multicoil2single(gt, maps)
#     #gt_single = fastmri.rss(network_utils.chans_to_coils(gt), dim=1)
#     gt_mag = fastmri.complex_abs(gt_single).cpu().numpy()
#     if complex:
#         gts.append(gt_single.cpu().numpy())
#     else:
#         gts.append(gt_mag)
#
#     # Get our mean image magnitude
#     # mean_mag = fastmri.complex_abs(samples[0].mean(dim=0).permute(1, 2, 0)).unsqueeze(0).cpu().numpy()
#     for i in range(6):
#         if complex:
#             mean_mag = samples[0][0:(2 ** i)].mean(dim=0).permute(1, 2, 0).unsqueeze(0).cpu().numpy()
#         else:
#             mean_mag = fastmri.complex_abs(samples[0][0:(2 ** i)].mean(dim=0).permute(1, 2, 0)).unsqueeze(0).cpu().numpy()
#         preds[i].append(mean_mag)
#
# psnrs = []
# for i in range(6):
#     report = calc_metrics(preds[i], gts, is_complex=complex)
#     psnrs.append(report['results']['mean_psnr'])
#
# # Plot the theoretical vs experiment PSNR vs num samples
# num = [1,2,4,8,16,32]
# vals = np.arange(1,num[-1],1)
# #theoretical = psnrs[-1] - (2.877 - 10*np.log10((2*vals)/(vals+1)))
# theoretical = psnrs[0] + 10*np.log10((2*vals)/(vals+1))
#
# plt.figure(figsize=(5,2))
# plt.plot(vals,theoretical, '--', label='Theoretical')
# plt.plot(num,psnrs, label='Experimental')
# plt.xlabel('Number of Samples')
# plt.ylabel('PSNR')
# plt.xscale('log')
# plt.xlim([1,32])
# #plt.title('PSNR vs Number of Samples')
# plt.legend()
# plt.grid(axis='both', which='both')
# plt.tight_layout()
# plt.show()
#
# np.save(os.path.join(eval_dir, 'psnrs{0}.npy'.format('complex' if complex else 'normal')), psnrs)

# #%% Get the FID
#
# num_samps = 8
# model.freq_sep = True
# temp = 1.0
#
# #Get a subset of the validation set
# #idxs = torch.randperm(len(data.val))[0:2376].numpy()
# #val_sub = torch.utils.data.Subset(data.val, idxs)
# val_sub = torch.utils.data.ConcatDataset([data.train, data.val])
# val_subloader= torch.utils.data.DataLoader(val_sub, batch_size = 16, shuffle=False)
#
# #VGG16 Model for embedding
# vgg16 = fid.VGG16Embedding()
# cfid_metric = fid.CFIDMetric(model,
#                           data.val_dataloader(), #data.test_dataloader(),
#                           vgg16,
#                           vgg16,
#                           mri_type = 'brain',
#                           resolution = input_dims[-1],
#                           cuda=True,
#                           num_samps=num_samps,
#                           model_type = 'cinn',
#                           challenge = 'multicoil',
#                           unet_cond=False,
#                           temp=temp,
#                           )
#
#
# y_predict, x_true, y_true = cfid_metric._get_generated_distribution()
# x_true_inter = torch.repeat_interleave(x_true, num_samps, dim=0)
# y_true_inter = torch.repeat_interleave(y_true, num_samps, dim=0)
# cfid_val = cfid_metric.get_cfid(y_predict, x_true_inter, y_true_inter)
# fid_val = cfid_metric.get_fid(y_predict, data.train_dataloader())
#
# print('cFID: {}'.format(cfid_val))
# print('FID: {}'.format(fid_val))
#
#
# #Get the path to save the metric scores
# report_path = os.path.join(model_path, 'fid')
# Path(report_path).mkdir(parents=True, exist_ok=True)
# report_dict = {'Metrics': {'cFID': cfid_val,
#                             'FID': fid_val},
#                     }
# report_file_path =  os.path.join(report_path, 'fid_report{0}.yaml'.format(temp))
# with open(report_file_path, 'w') as file:
#     documents = yaml.dump(report_dict, file)


#%% Get the FID
#
# num_samps = 1
# model.freq_sep = True
# temp = 1.0
# model.cuda()
# model.eval()
#
# #Get a subset of the validation set
# #idxs = torch.randperm(len(data.val))[0:2376].numpy()
# #val_sub = torch.utils.data.Subset(data.val, idxs)
# val_sub = torch.utils.data.ConcatDataset([data.train, data.val])
# val_subloader= torch.utils.data.DataLoader(val_sub, batch_size = 16, shuffle=False)
#
# miniset = torch.utils.data.Subset(data.test, [0,1])
# val_subloader= torch.utils.data.DataLoader(miniset, batch_size = 2, shuffle=False)
#
#
# #VGG16 Model for embedding
# vgg16 = fid.VGG16Embedding()
# cfid_metric = fid.CFIDMetric(model,
#                           data.test_dataloader(), #data.val_dataloader(), #data.test_dataloader(),
#                           vgg16,
#                           vgg16,
#                           mri_type = 'brain',
#                           resolution = input_dims[-1],
#                           cuda=True,
#                           num_samps=num_samps,
#                           model_type = 'cinn',
#                           challenge = 'multicoil',
#                           unet_cond=False,
#                           temp=temp,
#                           )
#
# #y_predict, x_true, y_true = cfid_metric._get_generated_distribution_map()
# y_predict, x_true, y_true = cfid_metric._get_generated_distribution()
# x_true_inter = torch.repeat_interleave(x_true, num_samps, dim=0)
# y_true_inter = torch.repeat_interleave(y_true, num_samps, dim=0)
# cfid_val = cfid_metric.get_cfid(y_predict, x_true_inter, y_true_inter)
# fid_val = cfid_metric.get_fid(y_predict, None)#data.train_dataloader())
#
# print('cFID: {}'.format(cfid_val))
# print('FID: {}'.format(fid_val))
#
#
# #Get the path to save the metric scores
# report_path = os.path.join(model_path, 'fid')
# Path(report_path).mkdir(parents=True, exist_ok=True)
# report_dict = {'Metrics': {'cFID': cfid_val,
#                             'FID': fid_val},
#                     }
# report_file_path =  os.path.join(report_path, 'fid_report{0}.yaml'.format(temp))
# with open(report_file_path, 'w') as file:
#     documents = yaml.dump(report_dict, file)

'''
with torch.no_grad():
    #x = batch[1].to(model.device)
    #c = batch[0].to(model.device) # TODO: Should be 0 for most cases
    x = data.train[100][1].unsqueeze(0).to(model.device)
    c = data.train[100][0].unsqueeze(0).to(model.device)
    maps = np.expand_dims(data.train[100][2], axis=0)
    masks = data.train[100][3]
    std = torch.tensor(data.train[100][4]).unsqueeze(0)

    #mmse = model.unet.reconstruct(c, mask = masks.to(model.device), maps = maps, std=std).to(model.device).permute(0,3,1,2)/std.reshape(-1,1,1,1).to(model.device)
    #mmse = mmse.float()
    
    cond = model.bij_cnet(c)
    preprocess = model.bij_cnet.preprocessing_net(c)
    

colormap = None
num = 0
    
viz.show_img(preprocess[num].unsqueeze(1), nrow=8, colormap=colormap)
viz.show_img(cond[0][num].unsqueeze(1), nrow=8, colormap=colormap)
viz.show_img(cond[1][num].unsqueeze(1), nrow=8, colormap=colormap)
viz.show_img(cond[2][num].unsqueeze(1), nrow=8, colormap=colormap)
viz.show_img(cond[3][num].unsqueeze(1), nrow=8, colormap=colormap)
viz.show_img(cond[4][num].unsqueeze(1), nrow=8, colormap=colormap)
viz.show_img(cond[5][num].unsqueeze(1), nrow=8, colormap=colormap)
'''

# %% Evaluate the training procedure to see where the bottleneck is
'''
import time
from tqdm import tqdm
loader = data.train_dataloader()
model.train()
model.cuda()

start = time.time()
for i, batch in enumerate(tqdm(loader)):
    #end = time.time()
    #batch = next(iter(loader))
    
    #print('Dataloading Time: {}'.format(end-start))
    start = time.time()

    out = model.training_step(batch, i)
    end = time.time()
    print('Training Step Time: {}'.format(end-start))
'''
# %%
# get_examples(model, model_path, data.val)

# %% Look for the max and min values of the dataset
'''
from tqdm import tqdm
max_val = 0
min_val = 0
for i, batch in enumerate(tqdm(data.val_dataloader())):
    x = network_utils.multicoil2single(batch[1], batch[2])
    
    if x.max() > max_val:
        max_val = x.max()
        
    if x.min() < min_val:
        min_val = x.min()
        
print(max_val)
print(min_val)
'''

# %%
'''
with torch.no_grad():
    #x = batch[1].to(model.device)
    #c = batch[1].to(model.device) # TODO: Should be 0 for most cases
    x = data.val[100][1].unsqueeze(0).to(model.device)
    c = data.val[100][0].unsqueeze(0).to(model.device)
    maps = data.val[100][2]
    masks = data.val[100][3].to(model.device)
    std = torch.tensor(data.val[100][4]).unsqueeze(0).to(model.device)
    
    if model.freq_sep:
        x_mask = network_utils.apply_mask(x, masks, std=std, inv_mask = True)
    
    z, ldj = model(x_mask,c, rev=False, mask= masks, maps = maps, std= std)
    
    z_zf, ldj_zf = model(c,c,rev=False, mask= masks, maps = maps, std= std)
    
    start = 0
    end = z.shape[1]/2
    add = z.shape[1]/4

    
    for i in range(5):
        z_mod = z*1.0
        z_mod[:, int(start):int(end)] = torch.randn_like(z_mod[:, int(start):int(end)])
        #z_mod[:, int(start):int(end)] = z_zf[:, int(start):int(end)]
        #z_mod = torch.randn_like(z)
        #z_mod[:, int(start):int(end)] = z[:, int(start):int(end)]
        
        x_hat, ldj_hat = model(z_mod, c, rev=True, mask= masks, maps = maps, std= std)
        recon, _ = network_utils.get_dc_image(x_hat, c, mask = masks, std = std)
        recon = network_utils.coils_to_chans(recon)
        
        viz.show_multicoil_combo(recon.detach().cpu(),maps)
        
        start = end * 1
        print(end)
        end = end + add
        
        #print(add)
        
        if i < 2:
            add = add/2
'''
