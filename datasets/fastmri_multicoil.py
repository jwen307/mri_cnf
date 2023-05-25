#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fastmri_multicoil.py
    - Contains the PL data module for multicoil fastMRI data
    - Contains helper functions for the data
"""


import torch
import pytorch_lightning as pl
import os
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import fastmri
from fastmri.data.transforms import to_tensor, tensor_to_complex_np




import sys
sys.path.append("..")

from util import viz, network_utils
from datasets.masks.mask import get_mask, apply_mask
from datasets.fastmri_multicoil_preprocess import preprocess_data, ImageCropandKspaceCompression, retrieve_metadata




# PyTorch Lightning Data Module for multicoil fastMRI data
class FastMRIDataModule(pl.LightningDataModule):

    def __init__(self, base_path, batch_size: int = 32, num_data_loader_workers: int = 4, **kwargs):
        """
        Initialize the data module for the fastMRI dataset.

        Parameters
        ----------
        base_path : str
            Location of the dataset (Ex: "/storage/fastMRI_brain/data/")
        batch_size : int, optional
            Size of a mini batch.
            The default is 4.
        num_data_loader_workers : int, optional
            Number of workers.
            The default is 8.

        Returns
        -------
        None.

        """
        super().__init__()

        self.base_path = base_path
        self.batch_size = batch_size
        self.num_data_loader_workers = num_data_loader_workers

        self.accel_rate = kwargs['accel_rate']
        self.img_size = kwargs['img_size']
        self.use_complex = kwargs['complex']
        kwargs['challenge'] = 'multicoil' # Only use the multicoil data. Combine for the single coil case
        self.challenge = kwargs['challenge']


        #Number of virtual coils
        self.num_vcoils = kwargs['num_vcoils']


        # Define a subset of scan types to use
        if 'scan_type' in kwargs:
            self.scan_type = kwargs['scan_type']
        else:
            self.scan_type = None

        # Define a subset of slices to use
        if 'slice_range' in kwargs:
            self.slice_range = kwargs['slice_range']
        else:
            self.slice_range = None

        # Define the type of mri image
        if 'mri_type' in kwargs:
            self.mri_type = kwargs['mri_type']
        else:
            self.mri_type = 'knee'

        # Preprocess the data if not done so already
        preprocess_data(self.base_path, **kwargs)

    def prepare_data(self):
        """
        Preparation steps like downloading etc.
        Don't use self here!

        Returns
        -------
        None.

        """
        None

    def setup(self, stage: str = None):
        """
        This is called by every GPU. Self can be used in this context!

        Parameters
        ----------
        stage : str, optional
            Current stage, e.g. 'fit' or 'test'.
            The default is None.

        Returns
        -------
        None.

        """



        train_dir = os.path.join(self.base_path, '{0}_{1}'.format('multicoil', 'train'))
        val_dir = os.path.join(self.base_path, '{0}_{1}'.format('multicoil', 'val'))

        max_val_dir_train = os.path.join(self.base_path,
                                         '{0}_{1}_{2}coils_preprocess'.format('multicoil', 'train', self.num_vcoils))
        max_val_dir_val = os.path.join(self.base_path,
                                       '{0}_{1}_{2}coils_preprocess'.format('multicoil', 'val', self.num_vcoils))

        # Assign train/val datasets for use in dataloaders
        self.train = MulticoilDataset(train_dir,
                                      max_val_dir_train,
                                      self.img_size, self.mri_type,
                                      self.accel_rate, self.scan_type,
                                      self.num_vcoils,
                                      self.slice_range
                                      )

        # Limit the slice range for the evaluation to remove edge slices with poor SNR
        if self.mri_type == 'brain':
            self.slice_range = [0,6]
        elif self.mri_type == 'knee':
            self.slice_range = 0.8

        self.val = MulticoilDataset(val_dir,
                                    max_val_dir_val,
                                    self.img_size, self.mri_type,
                                    self.accel_rate, self.scan_type,
                                    self.num_vcoils,
                                    self.slice_range
                                    )

        # Get a random subset of slices for testing but save them for later
        test_idx_file = os.path.join(self.base_path, 'test_idxs.npy')
        if os.path.exists(test_idx_file):
            self.test_idxs = np.load(test_idx_file)
        else:
            self.test_idxs = torch.randperm(len(self.val))[0:72].numpy()
            np.save(test_idx_file, self.test_idxs)

        self.test = torch.utils.data.Subset(self.val, self.test_idxs)


    def train_dataloader(self):
        """
        Data loader for the training data.

        Returns
        -------
        DataLoader
            Training data loader.

        """
        return DataLoader(self.train, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=True, pin_memory=True)

    def val_dataloader(self):
        """
        Data loader for the validation data.

        Returns
        -------
        DataLoader
            Validation data loader.

        """
        return DataLoader(self.val, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)

    def test_dataloader(self):
        """
        Data loader for the test data.

        Returns
        -------
        DataLoader
            Test data loader.

        """
        return DataLoader(self.test, batch_size=self.batch_size,
                          num_workers=self.num_data_loader_workers,
                          shuffle=False, pin_memory=True)


# Dataset class for multicoil MRI data
class MulticoilDataset(torch.utils.data.Dataset):
    def __init__(self, root, max_val_dir, img_size=320, mask_type='s4', accel_rate=4, scan_type=None, num_vcoils=8,
                 slice_range=None, **kwargs):
        '''
        scan_type: None, 'CORPD_FBK', 'CORPDFS_FBK' for knee
        scan_type: None, 'AXT2' for brain
        '''

        self.root = root
        self.img_size = img_size
        self.mask_type = mask_type
        self.accel_rate = accel_rate
        self.max_val_dir = max_val_dir
        self.examples = []

        self.multicoil_transf = MulticoilTransform(mask_type=self.mask_type,
                                                   img_size=self.img_size,
                                                   accel_rate=self.accel_rate,
                                                   num_vcoils=num_vcoils,
                                                   )

        self.slice_range = slice_range

        files = list(Path(root).iterdir())

        print('Loading Data')
        for fname in tqdm(sorted(files)):

            # Skip non-data files
            if fname.name[0] == '.':
                continue

            # Recover the metadata
            metadata, num_slices = retrieve_metadata(fname)

            with h5py.File(fname, "r") as hf:

                # Get the attributes of the volume
                attrs = dict(hf.attrs)
                attrs.update(metadata)

                # Get volumes of the specific scan type
                if scan_type is not None:
                    if attrs["acquisition"] != scan_type or attrs['encoding_size'][1] < img_size or hf['kspace'].shape[
                        1] <= num_vcoils:
                        continue

                # Use all the slices if a range is not specified
                if self.slice_range is None:
                    num_slices = hf['kspace'].shape[0]
                    slice_range = [0, num_slices]
                else:
                    if type(self.slice_range) is list:
                        slice_range = self.slice_range
                    elif self.slice_range < 1.0:
                        num_slices = hf['kspace'].shape[0]
                        # Use percentage of center slices (i.e. center 80% of slices)
                        slice_range = [int(num_slices * (1 - self.slice_range)), int(num_slices * self.slice_range)]

            self.examples += [(fname, slice_ind) for slice_ind in range(slice_range[0], slice_range[1])]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice = self.examples[i]

        # Get the normalizing value and vh
        with h5py.File(os.path.join(self.max_val_dir, fname.name), 'r') as hf:
            max_val = hf.attrs['max_val']
            vh = hf['vh'][dataslice]

        with h5py.File(fname, "r") as hf:
            # Get the compressed target kspace
            kspace = hf['kspace'][dataslice]

            acquisition = hf.attrs['acquisition']

            zf_img, gt_img, mask = self.multicoil_transf(kspace=kspace,
                                                         max_val=max_val,
                                                         vh=vh)

            zf_img = zf_img.squeeze(0)
            gt_img = gt_img.squeeze(0)

        return (
            zf_img.float(),
            gt_img.float(),
            mask,
            #np.float32(max_val),
            torch.tensor(max_val).float(),
            acquisition,
            fname.name,
            dataslice,
        )

# Transform for the multicoil dataset
class MulticoilTransform:

    def __init__(self, mask_type=None, img_size=320, accel_rate=4, num_vcoils=8):
        self.mask_type = mask_type
        self.img_size = img_size
        self.accel_rate = accel_rate
        self.num_vcoils = num_vcoils

    def __call__(self, kspace, max_val, vh):
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            max_val: the normalization value
            vh: the SVD component needed for coil compression
        """

        # kspace is dimension [num_coils, size0, size1, 2]
        kspace = to_tensor(kspace)

        # Compress to virtual coils
        gt_img, gt_k = get_compressed(kspace, self.img_size, num_vcoils=self.num_vcoils, vh=vh)

        # Stack the coils and real and imaginary
        gt_img = to_tensor(gt_img).permute(2, 3, 0, 1).reshape(-1, self.img_size, self.img_size).unsqueeze(0)

        # Apply the mask
        mask = get_mask(accel=self.accel_rate, size=self.img_size, mask_type=self.mask_type)
        masked_kspace = apply_mask(gt_k, mask)

        # Get the zf imgs
        masked_img = fastmri.ifft2c(masked_kspace).permute(0, 3, 1, 2).reshape(-1, self.img_size,
                                                                               self.img_size).unsqueeze(0)

        # Normalized based on the 95th percentile max value of the magnitude
        zf_img = masked_img / max_val
        gt_img = gt_img / max_val

        return zf_img, gt_img, mask




# Function to get the compressed image and kspace
def get_compressed(kspace: np.ndarray, img_size, num_vcoils = 8, vh = None):
    # inverse Fourier transform to get gt solution
    gt_img = fastmri.ifft2c(kspace)
    
    compressed_imgs = []

    #Compress to 8 virtual coils and crop
    compressed_img = ImageCropandKspaceCompression(tensor_to_complex_np(gt_img).transpose(1,2,0),
                                                   img_size, num_vcoils, vh)

        
    #Get the kspace for compressed imgs
    compressed_k = fastmri.fft2c(to_tensor(compressed_img).permute(2,0,1,3))
    
    
    return compressed_img, compressed_k






    

# Example usage
if __name__ == '__main__':


    kwargs = {
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
            }

    # Location of the dataset
    if kwargs['mri_type'] == 'brain':
        base_dir = "/storage/fastMRI_brain/data/"
    elif kwargs['mri_type'] == 'knee':
        base_dir = "/storage/fastMRI/data/"
    else:
        raise Exception("Please specify an mri_type in config")
    
    data = FastMRIDataModule(base_dir, batch = 4, **kwargs)
    data.prepare_data()
    data.setup()
    #dataset = MulticoilDataset(dataset_dir, max_val_dir, img_size, mask_type)
    img = data.val[0][1]
    mask = data.val[0][2]
    norm_val = data.val[0][3]

    viz.show_img(img.unsqueeze(0),rss=True)
    
