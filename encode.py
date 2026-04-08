"""
Copyright Snap Inc. 2021. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""
import os
import sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import math
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from torchvision.transforms import CenterCrop
from PIL import Image
from sync_batchnorm import DataParallelWithCallback

from modules.region_predictor import RegionPredictor

from animate import get_animation_region_params
import matplotlib
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
import torch.nn.functional as F
matplotlib.use('Agg')

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    region_predictor = RegionPredictor(num_feature=config['model_params']['num_feature'],
                                       num_channels=config['model_params']['num_channels'],
                                       **config['model_params']['region_predictor_params'])
    if not cpu:
        region_predictor.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    region_predictor.load_state_dict(checkpoint['region_predictor'])


    if not cpu:
        region_predictor = DataParallelWithCallback(region_predictor)
 
    region_predictor.eval()


    return  region_predictor


def encoding(source_image, driving_video, region_predictor, opt, cpu=False):
    qp_dir = opt.qp_dir
    quant_factor = opt.quant_factor
    cpu = opt.cpu

    with torch.no_grad():
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
 
        source_region_params = region_predictor(source)
        last_param = torch.cat([source_region_params['coeff'], source_region_params['bias']], dim=1).cpu().numpy() 

        for frame_idx in tqdm(range(1, driving.shape[2])):
            nf_dir = os.path.join(qp_dir, str(frame_idx+1).zfill(4))
            os.makedirs(nf_dir, exist_ok=True)
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            driving_region_params = region_predictor(driving_frame)
            param = torch.cat([driving_region_params['coeff'], driving_region_params['bias']], dim=1).cpu().numpy() 
               
            res_param = param - last_param
            # print(res_coeff.shape)
            res_param = (res_param*quant_factor).astype(np.int8)
            res_param = list(res_param.reshape(-1))
            param_bin_path=os.path.join(nf_dir,'param.bin')
            final_encoder_expgolomb(res_param,param_bin_path)  
            param_dec = final_decoder_expgolomb(param_bin_path)
            param_dec = data_convert_inverse_expgolomb(param_dec)   
            param_dec = np.array(param_dec).reshape(1,40,1,1)
            param_dec = last_param + param_dec/quant_factor
            last_param = param_dec


def main(opt):
    # opt.cpu = True
    # source_image = imageio.imread(opt.source_image)
    source_image_pil = Image.open(opt.source_image)
    source_image_w,  source_image_h = source_image_pil.size
    if source_image_w > source_image_h:
        source_image_pil = CenterCrop(source_image_h)(source_image_pil)

    else:
        source_image_pil = CenterCrop(source_image_w)(source_image_pil)
    
    source_image = np.array(source_image_pil)/255

    if opt.driving_video.endswith('.mp4'):
        reader = imageio.get_reader(opt.driving_video)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
    elif os.path.isdir(opt.driving_video):
    
        driving_video = []
        driving_frames = sorted(os.listdir(opt.driving_video))[: opt.n_frames]
        for driving_frame in driving_frames:
            driving_frame_path = os.path.join(opt.driving_video, driving_frame)

            driving_frame_pil = Image.open(driving_frame_path)
            driving_frame_w,  driving_frame_h = driving_frame_pil.size
            if driving_frame_w > driving_frame_h:
                driving_frame_pil = CenterCrop(driving_frame_h)(driving_frame_pil)
            else:
                driving_frame_pil = CenterCrop(driving_frame_w)(driving_frame_pil)
            driving_video.append(np.array(driving_frame_pil)/255)
    else:
        raise ValueError('Invalid driving video format')

    source_image = resize(source_image, opt.img_shape)[..., :3]
    driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
    region_predictor = load_checkpoints(config_path=opt.config,checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    encoding(source_image, driving_video, region_predictor, opt)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/ted-youtube384.yaml")
    parser.add_argument("--checkpoint", default='checkpoints/ted-youtube384.pth', help="path to checkpoint to restore")
    parser.add_argument("--source_image", default='/data/ysz/datasets/TEDTalk_dataset/00002/images/0020.png', help="path to source image")
    parser.add_argument("--driving_video", default='/data/ysz/datasets/TEDTalk_dataset/00002/images', help="path to driving video")
    parser.add_argument("--qp_dir")
   
    parser.add_argument("--n_frames", default=150, type=int)
    parser.add_argument("--quant_factor", default=127, type=int)
    parser.add_argument("--img_shape", default="384,384", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    main(parser.parse_args())
