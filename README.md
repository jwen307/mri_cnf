# A Conditional Normalizing Flow for Accelerated Multi-Coil MR Imaging

## Description
This is the official implementation of the paper "A Conditional Normalizing Flow for Accelerated Multi-Coil MR Imaging" (ICML 2023).
This repository contains the code for training and testing a conditional normalizing flow for multicoil MRI reconstruction.

## Installation
Please follow the instructions to setup the environment to run the repo.
1. Clone this repository
2. Create a new environment with the following commands
```
conda create -n mri_cnf python=3.9 numpy=1.23 pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge cupy cudatoolkit=11.8 cudnn cutensor nccl
conda install -c anaconda h5py=3.6.0
```
3. From the project root directory, install the requirements with the following command
```
pip install -r requirements.txt
```


## Usage Prerequisites
1. Download the fastMRI knee and brain datasets from [here](https://fastmri.org/)
2. Set the directory of the multicoil fastMRI knee and brain datasets to where they are stored on your device
    - Change [variables.py](variables.py) to set the paths to the dataset and your prefered logging folder
3. Change the configurations for training in the config files located in **train/configs/** 


## Training
First, set the directory to the **train** folder
```
cd train
```

To train our model, modify the configuration file in [config_cinn_unet_multicoil](train/configs/config_cinn_unet_multicoil.py) and run the following command
```
python train_cnf.py --model_type MulticoilCNF 
```

To run the baseline model proposed in ["Conditional Invertible Neural Networks for Medical Imaging"](https://arxiv.org/abs/2110.14520),
modify the configuration file in [config_cinn_unet_multicoil](train/configs/config_cinn_unet_multicoil.py) and run the following command
```
python train_cnf.py --model_type SinglecoilCNF 
```
Note: This can be run for the singlecoil case or for the multicoil case as in the ablation study.

All models will be saved in the logging folder specified in [variables.py](variables.py)

Some pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/1fhhdBS6LnOYvqmpAPkAovdr2gxZombKu?usp=sharing). 
Note: These models were retrained with the new, cleaned code found in this repo. Metrics are similar to those reported in the paper, but
may vary slightly.
## Evaluation
To get the evaluation metrics like PSNR, SSIM, FID, and cFID, run the following command
```
python eval_cnf.py --load_ckpt_dir <path to the model checkpoint>

#Example
python eval_cnf.py --load_ckpt_dir /home/user/mri_cnf/MulticoilCNF/version_0/
```

To generate reconstructions using a trained model, run the following command
```
python generate_imgs.py --load_ckpt_dir <path to the model checkpoint>
```
You can specify which image to generate, how many posterior samples to generate, and more in the script



## Notes
- The first time using a dataset will invoke the preprocessing step required for compressing the coils. 
Subsequent runs will be much faster since this step can be skipped.

## References
This work contains code that has been adapted from the following works:
```
@article{zbontar2018fastmri,
  title={fastMRI: An open dataset and benchmarks for accelerated MRI},
  author={Zbontar, Jure and Knoll, Florian and Sriram, Anuroop and Murrell, Tullie and Huang, Zhengnan and Muckley, Matthew J and Defazio, Aaron and Stern, Ruben and Johnson, Patricia and Bruno, Mary and others},
  journal={arXiv preprint arXiv:1811.08839},
  year={2018}
}

@article{devries2019evaluation,
  title={On the evaluation of conditional GANs},
  author={DeVries, Terrance and Romero, Adriana and Pineda, Luis and Taylor, Graham W and Drozdzal, Michal},
  journal={arXiv preprint arXiv:1907.08175},
  year={2019}
}

@misc{Ardizzone:github:18,
  author = {Ardizzone, Lynton and Bungert, Till and Draxler, Felix and Köthe, Ullrich and Kruse, Jakob and Schmier, Robert and Sorrenson, Peter},
  title = {{Framework for Easily Invertible Architectures (FrEIA)}},
  year = {2018},
  howpublished = {\url{https://github.com/vislearn/FrEIA}},
  note = {Accessed: 2022-11-05},
}

@article{Denker:JI:21,
  author={Denker, Alexander and Schmidt, Maximilian and Leuschner, Johannes and Maass, Peter},
  title={Conditional Invertible Neural Networks for Medical Imaging},
  journal= {J. Imaging},
  volume={7},
  number={11},
  pages={243},
  year={2021},
}

@journal{bendel2022arxiv,
  author = {Bendel, Matthew and Ahmad, Rizwan and Schniter, Philip},
  title = {A Regularized Conditional {GAN} for Posterior Sampling in Inverse Problems},
  year = {2022},
  journal={arXiv:2210.13389}
}

@misc{Seitzer2020FID,
  author={Maximilian Seitzer},
  title={{pytorch-fid: FID Score for PyTorch}},
  month={August},
  year={2020},
  note={Version 0.3.0},
  howpublished={\url{https://github.com/mseitzer/pytorch-fid}},
}

```

## Citation
Please cite our paper if you find this work useful
```
@article{Wen:ICML:23,
   author= {Wen, Jeffrey and Ahmad, Rizwan and Schniter, Philip}
   title = {A Conditional Normalizing Flow for Accelerated Multi-Coil MR Imaging},
   journal = {Proc. Int. Conf. Mach. Learn.},
   year={2023},
}
```