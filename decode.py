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
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sync_batchnorm import DataParallelWithCallback

from modules.generator import Generator
from modules.region_predictor import RegionPredictor
from animate import get_animation_region_params
import matplotlib
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *


matplotlib.use('Agg')

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")


def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.load(f)

    generator = Generator(num_regions=config['model_params']['num_regions'],
                          num_feature=config['model_params']['num_feature'],
                          num_channels=config['model_params']['num_channels'],
                          **config['model_params']['generator_params'])
    if not cpu:
        generator.cuda()

    region_predictor = RegionPredictor(num_feature=config['model_params']['num_feature'],
                                       num_channels=config['model_params']['num_channels'],
                                       **config['model_params']['region_predictor_params'])
    if not cpu:
        region_predictor.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    region_predictor.load_state_dict(checkpoint['region_predictor'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        region_predictor = DataParallelWithCallback(region_predictor)

    generator.eval()
    region_predictor.eval()

    return generator, region_predictor

def decode(source_image, generator, region_predictor, opt ):
    qp_dir = opt.qp_dir
    n_frames = opt.n_frames
    quant_factor = opt.quant_factor
    cpu = opt.cpu
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        
        if not cpu:
            source = source.cuda()
        
     
        source_region_params = region_predictor(source)
        last_param = torch.cat([source_region_params['coeff'], source_region_params['bias']], dim=1).cpu().numpy()
       
        for frame_idx in tqdm(range(1, n_frames)):
            nf_dir = os.path.join(qp_dir, str(frame_idx+1).zfill(4))

            param_bin_path=os.path.join(nf_dir,'param.bin')
            param_dec = final_decoder_expgolomb(param_bin_path)
            param_dec = data_convert_inverse_expgolomb(param_dec)   
            param_dec = np.array(param_dec).reshape(1,40,1,1)
            param_dec = last_param + param_dec/quant_factor
            last_param = param_dec

            param = torch.tensor(param_dec).type(torch.FloatTensor).to(source.device)
            coeff, bias = torch.split(param, [20, 20], dim=1)
            # print(coeff)
            
            driving_region_params = {
                'coeff': coeff,
                'bias': bias
            }
            
            out = generator(source, source_region_params=source_region_params, driving_region_params=driving_region_params)
            out_prediction = out['prediction'] * out['mask_pred'] + out['bg'] * (1 - out['mask_pred'])

            predictions.append(np.transpose(out_prediction.data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def main(opt):
    # opt.cpu = True
    source_image_pil = Image.open(opt.source_image)
    
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
        fps = opt.fps
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

    
    driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
    
 
    generator, region_predictor = load_checkpoints(config_path=opt.config,
                                                                checkpoint_path=opt.checkpoint, cpu=opt.cpu)

    predictions = decode(source_image, generator, region_predictor, opt)
    predictions.insert(0,source_image)
    frame_list = []
    frame_rgb_list = []
    for i, frame in enumerate(predictions):
        frame_grid = np.concatenate((source_image, driving_video[i], frame),axis=1)
        frame_rgb_list.append(frame)
        frame_list.append(img_as_ubyte(frame_grid))
    imageio.mimsave(os.path.join(opt.result_video, os.path.basename(opt.source_image)[5:-4]+'gen.mp4'), frame_list, fps=fps)
    outputs = []
    for x in frame_rgb_list:
    
        x = (x * 255).astype(np.uint8)
        r,g,b = np.split(x,3,axis=-1)
        outputs.append(r)
        outputs.append(g)
        outputs.append(b)
    with open(os.path.join(opt.result_video, os.path.basename(opt.source_image)[5:-4]+'gen.rgb'), 'wb') as f:
        f.writelines(np.ascontiguousarray(outputs))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/ted-youtube384.yaml")
    parser.add_argument("--checkpoint", default='checkpoints/ted-youtube384.pth', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='/data/ysz/datasets/TEDTalk_dataset/00002/images/0020.png', help="path to source image")
    parser.add_argument("--driving_video", default='/data/ysz/datasets/TEDTalk_dataset/00002/images', help="path to driving video")
    parser.add_argument("--result_video", default='./result.mp4', help="path to output")
    parser.add_argument("--n_frames", default=150, type=int)
    parser.add_argument("--fps", default=25, type=int)
    parser.add_argument("--qp_dir")
    parser.add_argument("--img_shape", default="384,384", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    parser.add_argument("--quant_factor", default=127, type=int)
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    main(parser.parse_args())
