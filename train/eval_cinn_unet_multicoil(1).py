#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 06:56:12 2022

@author: jeff
"""


import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers,seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
import os
import yaml
from pathlib import Path
import traceback
import numpy as np
import fastmri
import matplotlib.pyplot as plt

import sys
sys.path.append("..")

# TODO Change these for new models
from models.pl_models.multiscale_unet_multicoil import MultiscaleUnetMulticoil
from models.net_builds import build_multiscale_nets
from train.configs.config_cinn_unet_multicoil import Config
from evals import fid, metrics
from datasets.fastmri_multicoil import FastMRIDataModule
from util import helper, viz, network_utils, mail
import argparse
from evals.metrics import evaluate, evaluate_temps
from evals import fid
import time

# Location of the dataset
base_dir = "/storage/fastMRI/data/"
#base_dir= '/scratch/fastMRI/data/'
#base_dir = '/storage/data/'
#base_dir = '../../datasets/fastMRI/data'

'''
#Get the input arguments
args = util.bijflags()

train_net = args.train_net
load_ckpt_dir = args.load_ckpt_dir
load_ckpt_epoch = args.ckpt_epoch
load_ckpt_step = args.ckpt_step
'''
train_net = False
#load_ckpt_dir = 'None'
#load_ckpt_dir = '../logs/lightning_logs/version_28' #'../logs/default/version_3' or 'None' /storage/jeff/mri_cinn1/logs/
ckpt_dir = '/storage/jeff/mri_cinn1/logs/lightning_logs/'
load_version = 'version_{}'.format(80)
load_ckpt_dir = os.path.join(ckpt_dir,load_version)
eval_dir = os.path.join('../logs/lightning_logs', load_version)
#load_ckpt_dir='None'
load_ckpt_epoch = 49
#load_ckpt_step = 108100
load_ckpt_step = 76400

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def get_examples(model, model_path, dataset):
    
    #Specify the index of the images to test
    img_idxs = [25,100]

    model.cuda()
    model.eval()
    
    for img_idx in img_idxs:
        
        #Get the data
        cond = dataset[img_idx][0].unsqueeze(0)
        gt = dataset[img_idx][1].unsqueeze(0)
        #maps = np.expand_dims(dataset[img_idx][2], axis=0)

        masks = dataset[img_idx][3]
        std = torch.tensor(dataset[img_idx][4]).unsqueeze(0)

        maps = network_utils.get_maps(cond, model.acs_size, std)
        
        with torch.no_grad():
            #model.freq_sep = True
            #Get a few posterior samples
            samples, fsrecon = model.reconstruct(cond.to(model.device), 
                                        num_samples = 32,
                                        maps = maps, 
                                        mask = masks.to(model.device), 
                                        std = std.to(model.device),
                                        temp=1.0,
                                        give_fs = True,
                                        split_samples=True
                                        )
            '''
            if model.freq_sep:
                if torch.is_tensor(maps):
                    rep_maps = maps[0].unsqueeze(0).repeat(50,1,1,1)
                else:
                    rep_maps = np.repeat(np.expand_dims(maps[0],axis=0), 10, axis=0)   
                
                z = model.sample_distrib(num_samples = 50, temp = 1)
                outputs, _ = model(z, cond.repeat(50,1,1,1).to(model.device), rev=True,
                                   maps = rep_maps, 
                                   posterior_sample = False,
                                   mask= masks.to(model.device), 
                                   std= std[0].unsqueeze(0).repeat(10,1,1,1).to(model.device))
            '''
    
        #Save all the samples
        img_path = os.path.join(model_path, 'imgs')
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        
        val_range = None #(0,4.0)

        # Get the single coil magnitude ground truth
        gt = gt * std[0]
        gt_single = network_utils.multicoil2single(gt, maps)
        gt_mag = fastmri.complex_abs(gt_single)

        sample_mag = fastmri.complex_abs(samples[0][0].permute(1, 2, 0)).unsqueeze(0).cpu()
        mean_mag = fastmri.complex_abs(samples[0].mean(dim=0).permute(1, 2, 0)).unsqueeze(0).cpu()

        sample_psnr = recon_metrics1.psnr(gt_mag.numpy(), sample_mag.numpy())
        mean_psnr = recon_metrics1.psnr(gt_mag.numpy(), mean_mag.numpy())
        sample_ssim = recon_metrics1.ssim(gt_mag.numpy(), sample_mag.numpy())
        mean_ssim = recon_metrics1.ssim(gt_mag.numpy(), mean_mag.numpy())

        print(sample_psnr)
        print(mean_psnr)
        print(sample_ssim)
        print(mean_ssim)
            
        viz.show_img(samples[0],  val_range=val_range,
                     save_dir = os.path.join(img_path, 'posterior_samples{0}.png'.format(img_idx)))


        viz.show_multicoil_combo(gt, maps, val_range=val_range,
                     save_dir = os.path.join(img_path, 'gt{0}.png'.format(img_idx)))
        
        

        viz.show_multicoil_combo(cond[0], maps, val_range=val_range,
                     save_dir = os.path.join(img_path, 'zerofilled{0}.png'.format(img_idx)))
        
        
        
        viz.show_img(samples[0][0], val_range=val_range, 
                     save_dir = os.path.join(img_path, 'posterior_sample{0}.png'.format(img_idx)))
        
        
        
        viz.show_img(samples[0].mean(dim=0), val_range=val_range,
                     save_dir = os.path.join(img_path, 'sample_mean{0}.png'.format(img_idx)))
        
        '''
        if model.freq_sep:
            viz.show_multicoil_combo(outputs.mean(dim=0), maps, val_range = (0,1),
                     save_dir = os.path.join(img_path, 'output_mean{0}.png'.format(img_idx)))  
        '''
           
        #gt_single = network_utils.multicoil2single(gt, maps).permute(0,3,1,2)
        #gt_masked= network_utils.apply_mask(gt_single, masks, std=std, inv_mask = True)
        #viz.show_img(gt_masked)
        gt_masked= network_utils.apply_mask(gt, masks, std=std, inv_mask = True)
        viz.show_multicoil_combo(gt_masked, maps)
        gt_masked_single = network_utils.multicoil2single(gt_masked, maps)
        gt_masked_single_mag = fastmri.complex_abs(gt_masked_single)
        recon_fs = network_utils.multicoil2single(fsrecon[0], maps)
        recon_fs_mag = fastmri.complex_abs(recon_fs)
        viz.show_multicoil_combo(fsrecon[0], maps)
        viz.show_error_map(recon_fs_mag.cpu(),gt_masked_single_mag.cpu(), title='FS Error Map for cINN Posterior', limits=(0,0.5))
        
        cond_masked = network_utils.apply_mask(cond, masks, std=std, inv_mask = True)
        viz.show_multicoil_combo(cond_masked, maps)
        cond_masked_single = network_utils.multicoil2single(cond_masked, maps)
        cond_masked_single_mag = fastmri.complex_abs(cond_masked_single)
        viz.show_error_map(cond_masked_single_mag.cpu(),gt_masked_single_mag.cpu(), title='FS Error Map for UNet', limits=(0,0.5))
        
        
        
        unnorm_samples = samples[0] * std[0]
        capsd = unnorm_samples.std(dim=0).mean()
        #if not model.freq_sep:
        unnorm_samples = unnorm_samples.permute(0,2,3,1)
            
        unnorm_mag_samples = fastmri.complex_abs(unnorm_samples).unsqueeze(1)
        apsd = unnorm_mag_samples.std(dim=0).mean()
        viz.show_std_map(samples[0].cpu(), title='Standard Dev Map, cAPSD= {0:.3e}, APSD={1:.3e}'.format(capsd, apsd))
        
        #Look at the error between the mean and gt
        gt_single = network_utils.multicoil2single(gt, maps)
        gt_mag = fastmri.complex_abs(gt_single)
        #if not model.freq_sep:
        mean_mag = fastmri.complex_abs(samples[0].mean(dim=0).permute(1,2,0))
        #else:
        #mean_mag = fastmri.complex_abs(samples[0].mean(dim=0))
            
        viz.show_error_map(mean_mag.cpu().unsqueeze(0),gt_mag.cpu(), title='Error Map')
        
        plt.hist(gt_mag.flatten() - mean_mag.cpu().flatten(), bins=1000)
        plt.title('Error Histogram: {0:.3e}'.format(torch.sum(torch.abs(mean_mag.cpu().unsqueeze(0)-gt_mag.cpu()))))
        plt.legend()
        plt.show()
        
        
        
#%% Get the arguments for training
#if __name__ == "__main__":
    
#Use new configurations if not loading a pretrained model
if load_ckpt_dir == 'None':
    #Get the configurations
    config = Config()
    net_args = config.net_args
    ckpt=None
    
#Load the previous configurations
else:
    #Get the path to the checkpoint
    #ckpt = torch.load(os.path.join(load_ckpt_dir, 
    #                    'checkpoints', 
    #                    'epoch={0}.ckpt'.format(load_ckpt_epoch)))
    
    ckpt = os.path.join(load_ckpt_dir, 
                        'checkpoints', 
                        'epoch={0}-step={1}.ckpt'.format(load_ckpt_epoch, load_ckpt_step))
    
    #Get the configuration file
    config_file = os.path.join(load_ckpt_dir, 'config.pkl')
    net_args = util.read_pickle(config_file)
    
    #net_args['train_args']['lr'] = 2e-4




try:
    
    
    #Get the data
    data = FastMRIDataModule(base_dir, 
                             num_data_loader_workers=int(os.cpu_count()/4),
                             **net_args['train_args'],
                             )
    data.prepare_data()
    data.setup()

    #Setup the network
    if 'num_vcoils' in net_args['train_args']:
        in_ch = net_args['train_args']['num_vcoils'] * 2
    else:
        in_ch = 16
    
    #Set the input dimensions
    img_size = net_args['train_args']['img_size']
    input_dims = [in_ch, img_size, img_size]
    
    #Pick which build functions to use
    builds = [
                build_multiscale_nets.buildbij0,
                build_multiscale_nets.buildbij1,
                build_multiscale_nets.buildbij2,
                build_multiscale_nets.buildbij3,
                build_multiscale_nets.buildbij4,
                build_multiscale_nets.buildbij5,
                build_multiscale_nets.buildbij6,
                build_multiscale_nets.buildbij7,

              ]
    build_num = net_args['train_args']['build_num']
    
    #Monitor the learning rate
    seed_everything(1, workers=True)
    lr_monitor = LearningRateMonitor(logging_interval=None)
    
    
    #Load a checkpoint if specified
    if load_ckpt_dir != 'None':
        print("Loading previous checkpoint")
        #model.load_state_dict(ckpt['state_dict'])
        model = MultiscaleUnetMulticoil.load_from_checkpoint(ckpt, strict=False, input_dims= input_dims, 
                                                build_bij_func = builds[build_num], 
                                                net_args=net_args )
        
    else:
        #Initialize the network
        model = MultiscaleUnetMulticoil(input_dims, 
                         builds[build_num], 
                         net_args
                         )

    #Print the number of parameters
    print("Flow")
    network_utils.print_num_params(model.flow)
    print("Bijective Conditional")
    network_utils.print_num_params(model.bij_cnet)
    
    # Create the tensorboard logger
    #logger = loggers.TensorBoardLogger("../logs/")
    logger = loggers.TensorBoardLogger("/storage/jeff/mri_cinn1/logs/")

    

    if train_net:
        
        ckpt_callback = ModelCheckpoint(
            save_top_k = 3,
            monitor='val_bpd',
            mode = 'min',
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
            profiler='simple'
            #precision=64,
            #overfit_batches=0.01
            #detect_anomaly=True,
        )
        
        #Save the configurations
        model_path = trainer.logger.log_dir
        Path(model_path).mkdir(parents=True, exist_ok=True)
        config_file = os.path.join(model_path, 'config.pkl')
        util.write_pickle(net_args, config_file)
        
        if ckpt is None:
            print("Starting Training")
            trainer.fit(model, data.train_dataloader(), data.val_dataloader())
        
        else:
            print("Resuming Training")
            trainer.fit(model, data.train_dataloader(), data.val_dataloader(),ckpt_path=ckpt)
    
    else:
        model_path = load_ckpt_dir
    #Send an alert that the model is done training
    #mail.send_mail("Training Complete", "The training is complete")
    #get_examples(model, model_path, data.val)
    #model.freq_sep = True
    #evaluate(model, model_path, data, num_samples=50, model_type='cinn', temp = 1)
    #evaluate_temps(model, model_path, data, num_samples=[1,2,4,8,10,20,30], model_type='cinn', temps = [37])
    
    

except:
    #mail.send_mail("Training Error", 'Training has stopped')
    traceback.print_exc()


#%% Load the reconstructions and find the metrics
net_args['train_args']['slice_range']=0.8
#Get the data
data = FastMRIDataModule(base_dir,
                         num_data_loader_workers=int(os.cpu_count()/4),
                         **net_args['train_args'],
                         )
data.prepare_data()
data.setup()

#idxs = torch.randperm(len(data.val))[0:72].numpy()

#%%

from tqdm import tqdm
def calc_metrics(reconstructions, targets, is_complex=False):
    nmses = []
    psnrs = []
    ssims = []
    rsnrs = []
    for i in tqdm(range(len(reconstructions))):
        nmses.append(recon_metrics1.nmse(targets[i], reconstructions[i]))
        if is_complex:
            psnrs.append(recon_metrics1.psnr_complex(targets[i], reconstructions[i]))
        else:
            psnrs.append(recon_metrics1.psnr(targets[i], reconstructions[i]))
        ssims.append(recon_metrics1.ssim(targets[i], reconstructions[i]))
        rsnrs.append(recon_metrics1.rsnr(targets[i], reconstructions[i]))

    report_dict = {
        'results': {
            'mean_nmse': float(np.mean(nmses)),
            'std_err_nmse': float(np.std(nmses) / np.sqrt(len(nmses))),
            'mean_psnr': float(np.mean(psnrs)),
            'std_err_psnr': float(np.std(psnrs) / np.sqrt(len(psnrs))),
            'mean_ssim': float(np.mean(ssims)),
            'std_err_ssim': float(np.std(ssims) / np.sqrt(len(ssims))),
            'mean_rsnr (db)': float(np.mean(rsnrs)),
            'std_err_rsnr (db)': float(np.std(rsnrs) / np.sqrt(len(rsnrs))),
        }
    }


    return report_dict

#load_version = 'version_{}'.format(60)
#eval_dir = os.path.join('../logs/lightning_logs', load_version)

idxs = np.load('72_evals_idx.npy')

dataset = data.val
model = model.cuda()
model = model.eval()

# gts = []
# preds = []
#
# for k, samp_num in tqdm(enumerate(idxs)):
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
#                                              num_samples=8,
#                                              maps=maps,
#                                              mask=mask.to(model.device),
#                                              std=std.to(model.device),
#                                              temp=1.0,
#                                              give_fs=True,
#                                              split_samples=True,
#                                              #rss=True,
#                                              multicoil=True
#                                              )
#
#     # Get the single coil magnitude ground truth
#     gt = gt * std[0]
#     # gt_single = network_utils.multicoil2single(gt, maps)
#     #gt_single = fastmri.rss(network_utils.chans_to_coils(gt), dim=1)
#     #gt_mag = fastmri.complex_abs(gt_single).cpu().numpy()
#     gt_mag = fastmri.rss_complex(network_utils.chans_to_coils(gt), dim=1).cpu().numpy()
#     #gt_mag = fastmri.complex_abs(network_utils.multicoil2single(gt,maps)).cpu().numpy()
#     gts.append(gt_mag)
#
#     # Get our mean image magnitude
#     # mean_mag = fastmri.complex_abs(samples[0].mean(dim=0).permute(1, 2, 0)).unsqueeze(0).cpu().numpy()
#     #mean_mag = fastmri.complex_abs(samples[0].mean(dim=0)).unsqueeze(0).cpu().numpy()
#     #mean_mag = fastmri.rss_complex(samples[0].mean(dim=0)).unsqueeze(0).cpu().numpy()
#     mean_mag = fastmri.rss_complex(samples[0],dim=1).mean(dim=0).unsqueeze(0).cpu().numpy()
#     #mean_mag = fastmri.complex_abs(network_utils.multicoil2single(samples[0].mean(dim=0).unsqueeze(0),maps)).numpy()
#     #mean_mag = fastmri.complex_abs(network_utils.multicoil2single(samples[0], np.repeat(maps, 8, axis=0)).mean(dim=0)).unsqueeze(0).numpy()
#     preds.append(mean_mag)
#
#
#
# report_dict = calc_metrics(preds, gts, is_complex=False)
#
# report_path = os.path.join(eval_dir, 'psnr')
# Path(report_path).mkdir(parents=True, exist_ok=True)
#
# report_file_path = os.path.join(report_path, 'psnr_report{0}.yaml'.format(72))
# with open(report_file_path, 'w') as file:
#     documents = yaml.dump(report_dict, file)
#
# print(report_dict)








#%%
num_samps = 8
#model.freq_sep = True
temp = 1.0

#Get a subset of the validation set
#idxs = torch.randperm(len(data.val))[0:2376].numpy()
val_sub = torch.utils.data.Subset(data.val, idxs)
#val_sub = torch.utils.data.ConcatDataset([data.train, data.val])
val_subloader= torch.utils.data.DataLoader(val_sub, batch_size = 16, shuffle=False)

#VGG16 Model for embedding
vgg16 = fid.VGG16Embedding()
cfid_metric = fid.CFIDMetric(model,
                          val_subloader, #data.val_dataloader(),
                          vgg16,
                          vgg16,
                          resolution = input_dims[-1],
                          cuda=True,
                          num_samps=num_samps,
                          model_type = 'cinn',
                          challenge = 'multicoil',
                          unet_cond=False,
                          temp=temp,
                          rss=True,
                          mri_type = 'knee'
                          )


y_predict, x_true, y_true = cfid_metric._get_generated_distribution()
x_true_inter = torch.repeat_interleave(x_true, num_samps, dim=0)
y_true_inter = torch.repeat_interleave(y_true, num_samps, dim=0)
cfid_val = cfid_metric.get_cfid(y_predict, x_true_inter, y_true_inter)
fid_val = cfid_metric.get_fid(y_predict, y_true_inter)

print('cFID: {}'.format(cfid_val))
print('FID: {}'.format(fid_val))


#Get the path to save the metric scores
report_path = os.path.join(eval_dir, 'fid')
Path(report_path).mkdir(parents=True, exist_ok=True)
report_dict = {'Metrics': {'cFID': cfid_val,
                            'FID': fid_val},
                    }
report_file_path =  os.path.join(report_path, 'fid_report{0}.yaml'.format(temp))
with open(report_file_path, 'w') as file:
    documents = yaml.dump(report_dict, file)


# %% Try looking for the MAP estimate
dataset = data.val
img_idx = idxs[0]
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

#recon = model.reconstruct_MAPS(cond, maps, masks, std, epochs=500, rss=True)

with torch.no_grad():
    #model.freq_sep = True
    #Get a few posterior samples
    recon, fsrecon = model.reconstruct(cond.to(model.device),
                                num_samples = 1,
                                maps = maps,
                                mask = masks.to(model.device),
                                std = std.to(model.device),
                                temp=1.0,
                                give_fs = True,
                                split_samples=True,
                                multicoil=True
                                )

# Start at the mean image
# z = model.sample_distrib(1, temp=1.0).to(model.device)
# with torch.no_grad():
#     x, ldj = model(z, cond, rev=True, mask=masks, maps=maps, std=std[0])
#
# # Define the optimization
# x = x.requires_grad_(True)
# opt = torch.optim.Adam([x], lr=0.001)
#
# # Learning rate scheduler
# sched_factor = 0.9
# sched_patience = 10
# sched_thresh = 0.001
# sched_cooldown = 1
# sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     opt, factor=sched_factor,
#     patience=sched_patience, threshold=sched_thresh,
#     min_lr=1e-9, eps=1e-08, cooldown=sched_cooldown,
#     verbose=False)
#
#
#
# for i in range(epochs):
#     # Zero the gradient
#     opt.zero_grad()
#
#     # Find the latent vector
#     z, ldj = model(x, cond, rev=False, mask=masks, maps=maps, std=std[0])
#
#     # Get the loss
#     #loss = model.get_nll(z, ldj, give_bpd=True)
#     loss = (-ldj / (np.prod(z.shape[1:]) * np.log(2)) + 8) + torch.abs(z.norm() - 1280)
#     print('Epoch: {0}, Loss: {1}, Norm: {2}'.format(i, loss, z.detach().norm()))
#
#     # Backpropagate and update
#     loss.backward()
#     opt.step()
#
#     # Update the learning rate
#     sched.step(loss)

# Get the image
#x = x.detach().cpu()
#recon, _ = network_utils.get_dc_image(x, cond.cpu(), masks.cpu(), std=std)
#recon = recon * std
#mean_mag = fastmri.rss_complex(recon, dim=1)

mean_mag = fastmri.rss_complex(recon[0].detach().cpu(), dim=1)
viz.show_img(mean_mag)

# Get the single coil magnitude ground truth
gt = gt * std[0]
#gt_single = network_utils.multicoil2single(gt, maps)
#gt_mag = fastmri.complex_abs(gt_single)
gt_mag = fastmri.rss_complex(network_utils.chans_to_coils(gt), dim=1).cpu().numpy()

# Get the single coil magnitude condition
cond = cond * std[0]
#cond_single = network_utils.multicoil2single(cond, maps)
#cond_mag = fastmri.complex_abs(cond_single)
cond_mag = fastmri.rss_complex(network_utils.chans_to_coils(cond), dim=1).cpu().numpy()


# Get the magnitude image of the mean


# Save all the samples
#img_path = os.path.join(eval_dir, 'imgs/{}_map'.format(img_idx))
img_path = os.path.join(eval_dir, 'imgs/{}_post'.format(img_idx))
if not os.path.exists(img_path):
    os.makedirs(img_path)

viz.save_all_imgs(gt_mag, cond_mag, mean_mag, img_path, rotate=True)

#%% Get the FID

# num_samps = 1
# temp = 1.0
# model.cuda()
# model.eval()
#
# #Get a subset of the validation set
# val_sub = torch.utils.data.Subset(data.val, idxs)
# val_subloader= torch.utils.data.DataLoader(val_sub, batch_size = 16, shuffle=False)
#
#
# #VGG16 Model for embedding
# vgg16 = fid.VGG16Embedding()
# cfid_metric = fid.CFIDMetric(model,
#                           val_subloader, #data.val_dataloader(), #data.test_dataloader(),
#                           vgg16,
#                           vgg16,
#                           mri_type = 'knee',
#                           resolution = input_dims[-1],
#                           cuda=True,
#                           num_samps=num_samps,
#                           model_type = 'cinn',
#                           challenge = 'multicoil',
#                           unet_cond=False,
#                           temp=temp,
#                           rss=True
#                           )
#
# y_predict, x_true, y_true = cfid_metric._get_generated_distribution_map()
# #y_predict, x_true, y_true = cfid_metric._get_generated_distribution()
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
# report_path = os.path.join(eval_dir, 'fid')
# Path(report_path).mkdir(parents=True, exist_ok=True)
# report_dict = {'Metrics': {'cFID': cfid_val,
#                             'FID': fid_val},
#                     }
# report_file_path =  os.path.join(report_path, 'fid_report{0}.yaml'.format(temp))
# with open(report_file_path, 'w') as file:
#     documents = yaml.dump(report_dict, file)

#%% Look at PSNR vs Number of Samples
# idxs = np.load(eval_dir + '/72_evals_idx.npy')
#
# dataset = data.val
# model = model.cuda()
# model = model.eval()
# complex = True
#
# gts = []
# preds = [[] for i in range(6)]
# times = []
#
# for k, samp_num in tqdm(enumerate(idxs)):
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
#         start = time.time()
#         # Get a few posterior samples
#         samples, fsrecon = model.reconstruct(cond.to(model.device),
#                                              num_samples=32,
#                                              maps=maps,
#                                              mask=mask.to(model.device),
#                                              std=std.to(model.device),
#                                              temp=1.0,
#                                              give_fs=True,
#                                              split_samples=True,
#                                              #rss=True,
#                                              multicoil=True
#                                              )
#
#         end = time.time()
#
#         times.append(end-start)
#
#     # Get the single coil magnitude ground truth
#     gt = gt * std[0]
#     gt_single = network_utils.multicoil2single(gt, maps)
#     #gt_single = fastmri.rss(network_utils.chans_to_coils(gt), dim=1)
#     #gt_mag = fastmri.complex_abs(gt_single).cpu().numpy()
#     if complex:
#         gts.append(gt_single.cpu().numpy())
#     else:
#         gt_mag = fastmri.rss_complex(network_utils.chans_to_coils(gt), dim=1).cpu().numpy()
#         gts.append(gt_mag)
#
#     # Get our mean image magnitude
#     # mean_mag = fastmri.complex_abs(samples[0].mean(dim=0).permute(1, 2, 0)).unsqueeze(0).cpu().numpy()
#     for i in range(6):
#         if complex:
#             mean_mag = samples[0][0:(2 ** i)].mean(dim=0).unsqueeze(0)
#             mean_mag = network_utils.multicoil2single(mean_mag, maps).cpu().numpy()
#         else:
#             #mean_mag = fastmri.rss_complex(samples[0][0:(2 ** i)].mean(dim=0)).unsqueeze(0).cpu().numpy()
#             mean_mag = fastmri.rss_complex(samples[0],dim=1)[0:(2 ** i)].mean(dim=0).unsqueeze(0).cpu().numpy()
#             #mean_mag = fastmri.complex_abs(samples[0][0:(2 ** i)].mean(dim=0)).unsqueeze(0).cpu().numpy()
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


# #%% Save images
# rotate = True
# dataset = data.val
#
# model.cuda()
# model.eval()
#
# for img_idx in idxs[3:4]:
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
#                                     split_samples=True,
#                                     #rss=True
#                                     multicoil=True
#                                     )
#
#     #Save all the samples
#     img_path = os.path.join(eval_dir, 'imgs/{}'.format(img_idx))
#     if not os.path.exists(img_path):
#         os.makedirs(img_path)
#
#     # Get the single coil magnitude ground truth
#     gt = gt * std[0]
#     #gt_single = network_utils.multicoil2single(gt, maps)
#     #gt_single = fastmri.rss(network_utils.chans_to_coils(gt), dim=1)
#     #gt_mag = fastmri.complex_abs(gt_single)
#     gt_mag = fastmri.rss_complex(network_utils.chans_to_coils(gt), dim=1).cpu().numpy()
#
#     # Get the single coil magnitude condition
#     cond = cond * std[0]
#     #cond_single = network_utils.multicoil2single(cond, maps)
#     #cond_single = fastmri.rss(network_utils.chans_to_coils(cond), dim=1)
#     #cond_mag = fastmri.complex_abs(cond_single)
#     cond_mag = fastmri.rss_complex(network_utils.chans_to_coils(cond), dim=1).cpu().numpy()
#
#     #Get the magnitude image of the mean
#     #mean_mag = fastmri.complex_abs(samples[0].mean(dim=0).permute(1,2,0)).unsqueeze(0).cpu()
#     samples_mag = fastmri.rss_complex(samples[0], dim=1).cpu()
#     mean_mag = samples_mag.mean(dim=0).unsqueeze(0).cpu().numpy()
#
#     viz.save_all_imgs(gt_mag, cond_mag, mean_mag, img_path, rotate=rotate)
#
#     #Get the standard deviation map
#     #samples_mag = fastmri.complex_abs(samples[0].permute(0,2,3,1)).cpu()
#     samples_std = samples_mag.std(dim=0).unsqueeze(0)
#
#     viz.save_std_map(samples_std, colorbar=False, save_dir = img_path, rotate=rotate, limits=[0,2e-5])
#
#     #Save the posterior samples with zoomed in regions
#     rect_colors = ['deepskyblue', 'y']
#     rect = [[225,195,30,30],[255,240,30,30]] #[x-top right corner, y-top right corner, x-width, y-width] where x is right left, y is top down
#     viz.save_posteriors(samples_mag, rotate=rotate, save_dir = os.path.join(img_path, 'posteriors'),
#                         rect = rect, rect_colors = rect_colors, num_samples=10, val_range=(0,0.00075))
#     viz.save_posteriors(gt_mag, rotate=rotate, save_dir=os.path.join(img_path, 'gt'),
#                         rect=rect, rect_colors=rect_colors, val_range=(0,0.00075))
#     viz.save_posteriors(cond_mag, rotate=rotate, save_dir=os.path.join(img_path, 'cond'),
#                         rect=rect, rect_colors=rect_colors, val_range=(0,0.00075))
#     rect_colors = ['r', 'k']
#     viz.save_posteriors(samples_std, rotate=rotate, std= True, save_dir=os.path.join(img_path, 'std'),
#                         #val_range=(0,2e-5),
#                         rect=rect, rect_colors=rect_colors)
#%%
#np.save(os.path.join(model_path, 'nmse.npy'), np.array(nmses))
# #%% Check out interpolation
# # Get the data
# idx1 = 100
# cond1 = data.val[idx1][0].unsqueeze(0)
# gt1 = data.val[idx1][1].unsqueeze(0)
# maps1 = np.expand_dims(data.val[idx1][2], axis=0)
# masks1 = data.val[idx1][3]
# std1 = torch.tensor(data.val[idx1][4]).unsqueeze(0)
#
# # Get the data
# idx2 = 24
# cond2 = data.val[idx2][0].unsqueeze(0)
# gt2 = data.val[idx2][1].unsqueeze(0)
# maps2 = np.expand_dims(data.val[idx2][2], axis=0)
# masks2 = data.val[idx2][3]
# std2 = torch.tensor(data.val[idx2][4]).unsqueeze(0)
#
# cond12 = 0.5*cond1 + 0.5*cond2
# kspace1 = fastmri.ifft2c(network_utils.chans_to_coils(cond1)*std1)
# kspace2 = fastmri.ifft2c(network_utils.chans_to_coils(cond2)*std2)
# kspace12 = 0.5*kspace1 + 0.5*kspace2
# cond12_k = fastmri.fft2c(kspace12)
#
# gt_m = gt1*1.0
# gt_m[:,:, 180:220,180:220] = 0
#
# with torch.no_grad():
#     # mmse1, feats1 = model.unet(cond1.to(model.device), give_dc=model.unet.data_consistency, mask=masks1.to(model.device), std=std1, give_features=True)
#     # mmse2, feats2 = model.unet(cond2.to(model.device), give_dc=model.unet.data_consistency, mask=masks2.to(model.device), std=std1, give_features=True)
#     #
#     # feats12 = 0.5*feats1 + 0.5*feats2
#     # z = model.sample_distrib(num_samples=1)
#     # c = model.bij_cnet(feats12)
#     # x,_ = model.flow(z,c, rev=True)
#
#     # model.freq_sep = True
#     # See what happens if you pass different cond and gt
#     x = network_utils.apply_mask(gt1, masks1, std=std1, inv_mask=True)
#     z, ldj = model(x.to(model.device), cond1.to(model.device), mask=masks1.to(model.device), std=std1)
#     x = network_utils.apply_mask(gt_m, masks1, std=std1, inv_mask=True)
#     z1, ldj = model(x.to(model.device), cond1.to(model.device), mask = masks1.to(model.device), std=std1)
#     # Get a few posterior samples
#     # samples, fsrecon = model.reconstruct(cond1.to(model.device),
#     #                                      num_samples=32,
#     #                                      maps=maps1,
#     #                                      mask=masks1.to(model.device),
#     #                                      std=std1.to(model.device),
#     #                                      temp=1.0,
#     #                                      give_fs=True,
#     #                                      split_samples=True
#     #                                      )
#
# #%%
# thres = 8
# z1 = z1.cpu()
# z = z.cpu()
# diff = torch.abs(z1 - z)
#
# num_latents = torch.sum(diff>thres)
#
# z1[diff>thres] = torch.randn(num_latents)
#
# with torch.no_grad():
#     pred, _ = model(z1.to(model.device),cond1.to(model.device), rev=True, mask=masks1.to(model.device), std=std1)
#
# pred_full,_ = network_utils.get_dc_image(pred,cond1.to(model.device),masks1.to(model.device))
# viz.show_multicoil_combo(pred_full, maps1)
# viz.show_multicoil_combo(gt_m, maps1)
#
#
#
# #%% Look at the kspace std dev
# import matplotlib as mpl
# samples_all = torch.concat(samples,dim=0).cpu()
# kspace_all = fastmri.fft2c(samples_all.permute(0,2,3,1))
# kspace_std = kspace_all.std(dim=0)
#
# #Get the log of the magnitude
# magnitude= torch.log(fastmri.complex_abs(kspace_std))
#
# #Plot the magnitude
# plt.imshow(magnitude.cpu().numpy(), cmap = mpl.colormaps['viridis'])
# plt.colorbar(fraction=0.046, pad = 0.04)
# plt.show()
#
#
# # Look at GT with some noise
# gt_noise = gt1.repeat(32,1,1,1) + 0.001*torch.randn_like(gt1.repeat(32,1,1,1))
# rep_maps = np.repeat(maps1, 32, axis=0)
# gt_noise = network_utils.multicoil2single(gt_noise,rep_maps)
# gt_noise_k = fastmri.fft2c(gt_noise * std1)
# gt_noise_k_std = gt_noise_k.std(dim=0)
# gt_noise_k_mag = torch.log(fastmri.complex_abs(gt_noise_k_std))
# viz.show_map(gt_noise_k_mag.unsqueeze(0))
# #viz.plot_kspace(gt_noise_k[1], val_range = val_range)
#
# #%% Look at difference image between two posteriors
# diff = samples_all[0] - samples_all[1]
# viz.show_error_map(fastmri.complex_abs(samples_all[5].permute(1,2,0)).unsqueeze(0),fastmri.complex_abs(samples_all[15].permute(1,2,0)).unsqueeze(0))

#%% Get the PSNR vs number of samples


# # Specify the index of the images to test
# img_idx = 100
# dataset = data.val
#
# model.cuda()
# model.eval()
# num = [1,2,4,8,16,32]
# psnr_vals = []
#
#
# # Get the data
# cond = dataset[img_idx][0].unsqueeze(0)
# gt = dataset[img_idx][1].unsqueeze(0)
# maps = np.expand_dims(dataset[img_idx][2], axis=0)
# masks = dataset[img_idx][3]
# std = torch.tensor(dataset[img_idx][4]).unsqueeze(0)
#
# with torch.no_grad():
#     # model.freq_sep = True
#     # Get a few posterior samples
#     samples, fsrecon = model.reconstruct(cond.to(model.device),
#                                          num_samples=32,
#                                          maps=maps,
#                                          mask=masks.to(model.device),
#                                          std=std.to(model.device),
#                                          temp=1.0,
#                                          give_fs=True,
#                                          split_samples=True
#                                          )
#     # Get the single coil magnitude ground truth
#     gt = gt * std[0]
#     gt_single = network_utils.multicoil2single(gt, maps)
#     gt_mag = fastmri.complex_abs(gt_single)
#
# for n in num:
#
#     mean_mag = fastmri.complex_abs(samples[0][0:n].mean(dim=0).permute(1, 2, 0)).unsqueeze(0).cpu()
#
#     mean_psnr = recon_metrics1.psnr(gt_mag.numpy(), mean_mag.numpy())
#     psnr_vals.append(mean_psnr)
#
# # Plot the theoretical vs experiment PSNR vs num samples
# vals = np.arange(1,num[-1],1)
# theoretical = psnr_vals[-1] - (2.877 - 10*np.log10((2*vals)/(vals+1)))
#
# plt.plot(vals,theoretical,label='Theoretical')
# plt.plot(num,psnr_vals, label='Experimental')
# plt.xlabel('Number of Samples')
# plt.ylabel('PSNR')
# plt.title('PSNR vs Number of Samples')
# plt.legend()
# plt.show()

#%% Get the FID
# val_sub = torch.utils.data.Subset(data.val, idxs)
# #val_sub = torch.utils.data.ConcatDataset([data.train, data.val])
# val_subloader= torch.utils.data.DataLoader(val_sub, batch_size = 8, shuffle=False)
# num_samps = 8
#
# #VGG16 Model for embedding
# vgg16 = fid.VGG16Embedding()
# cfid_metric = fid.CFIDMetric(model,
#                          val_subloader, #data.val_dataloader(),
#                          vgg16,
#                          vgg16,
#                          mri_type = 'knee',
#                          resolution = 320,
#                          cuda=True,
#                          num_samps=num_samps,
#                          model_type = 'cinn',
#                          challenge = 'multicoil',
#                          unet_cond=True,
#                          temp=1.0,
#                          is_complex=True,
#                          rss = True
#                          )
#
# y_predict, x_true, y_true = cfid_metric._get_generated_distribution()
# x_true_inter = torch.repeat_interleave(x_true, num_samps, dim=0)
# y_true_inter = torch.repeat_interleave(y_true, num_samps, dim=0)
# cfid_val = cfid_metric.get_cfid(y_predict, x_true_inter, y_true_inter)
# fid_val = cfid_metric.get_fid(y_predict, data.train_dataloader())
#
# print('cFID: {}'.format(cfid_val))
# print('FID: {}'.format(fid_val))


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

#%% Evaluate the training procedure to see where the bottleneck is
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
#%%
#get_examples(model, model_path, data.val)

#%% Look for the max and min values of the dataset
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

#%%
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

#%%
#
# #Specify the index of the images to test
# img_idxs = [100]
#
# model.cuda()
# model.eval()
#
# dataset=data.val
#
# img_idx = 25
#
# #Get the data
# cond = dataset[img_idx][0].unsqueeze(0)
# gt = dataset[img_idx][1].unsqueeze(0)
# maps = np.expand_dims(dataset[img_idx][2], axis=0)
# masks = dataset[img_idx][3]
# std = torch.tensor(dataset[img_idx][4]).unsqueeze(0)
#
# with torch.no_grad():
#     #model.freq_sep = True
#     #Get a few posterior samples
#     samples, fsrecon = model.reconstruct(cond.to(model.device),
#                                 num_samples = 50,
#                                 maps = maps,
#                                 mask = masks.to(model.device),
#                                 std = std.to(model.device),
#                                 temp=1.0,
#                                 give_fs = True,
#                                 split_samples=True
#                                 )
#
#
#
# # Look at the GT kspace
# gt_single = network_utils.multicoil2single(gt, maps)
# gt_k = fastmri.fft2c(gt_single * std)
# magnitude = torch.log(fastmri.complex_abs(gt_k)+ 1e-9)
# val_range = (magnitude.min(), magnitude.max())
# viz.plot_kspace(gt_k.squeeze(0), val_range = val_range)
#
# # Look at GT with some noise
# gt_noise = gt_single.repeat(50,1,1,1) + 0.05*torch.randn_like(gt_single.repeat(50,1,1,1))
# gt_noise_k = fastmri.fft2c(gt_noise * std)
# gt_noise_k_std = gt_noise_k.std(dim=0)
# gt_noise_k_mag = fastmri.complex_abs(gt_noise_k_std)
# viz.show_map(gt_noise_k_mag.unsqueeze(0))
# viz.plot_kspace(gt_noise_k[1], val_range = val_range)
# #viz.show_std_map(gt_noise_k)
#
# #Add noise to the unknown parts of kspace and look at std dev maps
# noise =  0.05*torch.randn_like(gt_single.repeat(50,1,1,1))
# noise_k = fastmri.fft2c(noise * std)
# mask2D =  masks.permute(0,2,1).repeat(1,masks.shape[1],1)
# mask2D = mask2D*-1 + 1
# masked_noise_k = mask2D.unsqueeze(-1).repeat(1,1,1,2) * noise_k.cpu()
# viz.plot_kspace(masked_noise_k[0], val_range=val_range)
# masked_noise = fastmri.ifft2c(masked_noise_k)/std
# viz.show_img(masked_noise[0])
# gt_noise = gt_single + masked_noise
# gt_noise_k = fastmri.fft2c(gt_noise * std)
# gt_noise_k_std = gt_noise_k.std(dim=0)
# gt_noise_k_mag = fastmri.complex_abs(gt_noise_k_std)
# viz.show_map(gt_noise_k_mag.unsqueeze(0))
#
#
# #Look at the conditional kspace
# cond_single = network_utils.multicoil2single(cond, maps)
# cond_masked = network_utils.apply_mask(cond_single.permute(0,3,1,2), masks, std=std)
# cond_k = fastmri.fft2c(cond_masked.permute(0,2,3,1) * std)
# viz.plot_kspace(cond_k.squeeze(0), val_range = val_range)
# unet_k = fastmri.fft2c(cond_single*std)
# viz.plot_kspace(unet_k.squeeze(0), val_range = val_range)
#
#
# #Get the kspace maps for the samples
# recon_k = fastmri.fft2c(samples[0].permute(0,2,3,1))# * std.to(model.device))
# recon_mean_k = fastmri.fft2c(samples[0].mean(dim=0).permute(1,2,0) * std.to(model.device))
# viz.plot_kspace(recon_k[0].cpu(), val_range = val_range)
# viz.plot_kspace(recon_mean_k.cpu(), val_range = val_range)
#
# #Look at the std-dev of kspace maps
# recon_k_std = recon_k.std(dim=0).cpu()
# viz.plot_kspace(recon_k_std)
# viz.show_std_map(recon_k)
#
# recon_k_std_mag = fastmri.complex_abs(recon_k_std)
# viz.show_map(recon_k_std_mag.unsqueeze(0))

# %%
