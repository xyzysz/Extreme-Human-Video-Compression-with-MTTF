
import matplotlib

matplotlib.use('Agg')

import os
import sys
import yaml
from argparse import ArgumentParser
from time import gmtime, strftime, localtime
from shutil import copy

from frames_dataset import FramesDataset

from modules.generator import Generator
from modules.region_predictor import RegionPredictor

import torch
import numpy as np
from train import train
from SemanticGuidedHumanMatting.model.model import HumanMatting
import yaml

if __name__ == "__main__":

    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

    parser = ArgumentParser()
    parser.add_argument("--config", default='config/ted384.yaml', help="path to config")
    parser.add_argument("--mode", default="train", choices=["train"])
    parser.add_argument("--log_dir", default='log', help="path to log into")
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")
    parser.add_argument("--device_ids", default="0,1,2,3", type=lambda x: list(map(int, x.split(','))),
                        help="Names of the devices comma separated.")
    parser.add_argument("--verbose", dest="verbose", action="store_true", help="Print model architecture")
    parser.set_defaults(verbose=False)

    opt = parser.parse_args()
    with open(opt.config) as f:
        config = yaml.load(f,Loader = yaml.FullLoader)

    if opt.checkpoint is not None:
        log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
    else:
        log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
        log_dir += ' ' + strftime("%d_%m_%y_%H.%M.%S", localtime())

    generator = Generator(num_regions=config['model_params']['num_regions'],
                            num_feature=config['model_params']['num_feature'],
                          num_channels=config['model_params']['num_channels'],
                          num_background=config['model_params']['num_background'],
                          feature_size=config['model_params']['feature_size'],
                          num_down_blocks=int(np.log2(config['dataset_params']['frame_shape'][-1]/config['model_params']['feature_size'])),
                          **config['model_params']['generator_params'])

    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
    if opt.verbose:
        print(generator)

    region_predictor = RegionPredictor(num_feature=config['model_params']['num_feature'],
                                       num_channels=config['model_params']['num_channels'],
                                       feature_size=config['model_params']['feature_size'],
                                       **config['model_params']['region_predictor_params'])

    if torch.cuda.is_available():
        region_predictor.to(opt.device_ids[0])

    if opt.verbose:
        print(region_predictor)


    mask_supervisor = HumanMatting(backbone='resnet50')
    state_dict = torch.load('SemanticGuidedHumanMatting/pretrained/SGHM-ResNet50.pth', map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    mask_supervisor.load_state_dict(new_state_dict)
    mask_supervisor.eval().to(opt.device_ids[0])
    mask_supervisor.requires_grad_(False)

    dataset = FramesDataset(is_train=(opt.mode.startswith('train')), **config['dataset_params'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
        copy(opt.config, log_dir)

    if opt.mode == 'train':
        print("Training...")
        train(config, generator, region_predictor, mask_supervisor, opt.checkpoint, log_dir, dataset, opt.device_ids)
    
