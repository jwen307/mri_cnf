# -*- coding: utf-8 -*-

import argparse
import json
import pickle
from pathlib import Path

import sys
sys.path.append("../../")

from models.pl_models.multiscale_unet_multicoil import MulticoilCNF
from models.pl_models.multiscale_singlecoil import SinglecoilCNF

   
def flags():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_type',
        type=str,
        default='MulticoilCNF',
        help='Type of model to use. Options: MulticoilCNF, SinglecoilCNF'
    )
    
    parser.add_argument(
        '--load_ckpt_dir',
        type=str,
        default='None',
        help='Directory of the checkpoint that you want to load'
        )

    parser.add_argument(
        '--load_last_ckpt',
        action='store_true',
        help='Whether to load the last checkpoint or best val bpd checkpoint'
        )

    # parser.add_argument(
    #     '--ckpt_epoch',
    #     type=int,
    #     default=9,
    #     help='Epoch of the checkpoint that you want to load'
    #     )
    #
    # parser.add_argument(
    #     '--ckpt_step',
    #     type=int,
    #     default=9,
    #     help='Step of the checkpoint that you want to load'
    #     )


    return parser.parse_args()

# Read a json file
def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    
    
def read_pickle(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    
    return obj

def write_pickle(obj, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


# Function to get root of the project
def get_root() -> Path:
    return Path(__file__).parent.parent


# Function to load different models
def load_model(model_type, config, ckpt=None):
    '''
    model_type: Type of model to load
    config: Configuration file for the model
    ckpt: Checkpoint to load
    '''

    if model_type == 'MulticoilCNF':
        # Get the model
        model = MulticoilCNF(config)
    elif model_type == 'SinglecoilCNF':
        model = SinglecoilCNF(config)
    else:
        raise ValueError('Model type not recognized')

    # Load the checkpoint if needed
    if ckpt is not None:
        model.load_from_checkpoint(ckpt)

    return model


# Function to select config file
def select_config(model_type):
    if model_type == 'MulticoilCNF':
        from train.configs.config_cinn_unet_multicoil import Config
    elif model_type == 'SinglecoilCNF':
        from train.configs.config_singlecoil import Config
    else:
        raise ValueError('Model type not recognized')

    return Config()