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

from modules.generator import Generator
from modules.region_predictor import RegionPredictor
from animate import get_animation_region_params
import matplotlib

from flow_visual import *
from modules.util import make_coordinate_grid
import torch.nn.functional as F

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


def make_animation(source_image, driving_video, generator, region_predictor, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        source_region_params = region_predictor(source)
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            driving_region_params = region_predictor(driving_frame)
                
            out = generator(source, source_region_params=source_region_params, driving_region_params=driving_region_params)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


def main(opt):
    source_image_pil = Image.open(opt.source_image)
    source_image_w,  source_image_h = source_image_pil.size
    if source_image_w > source_image_h:
        source_image_pil = CenterCrop(source_image_h)(source_image_pil)

    else:
        source_image_pil = CenterCrop(source_image_w)(source_image_pil)
    
    source_image = np.array(source_image_pil)

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
            driving_video.append(np.array(driving_frame_pil))
    else:
        raise ValueError('Invalid driving video format')

    source_image = resize(source_image, opt.img_shape)[..., :3]
    driving_video = [resize(frame, opt.img_shape)[..., :3] for frame in driving_video]
    generator, region_predictor = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, cpu=opt.cpu)
    predictions = make_animation(source_image, driving_video, generator, region_predictor,cpu=opt.cpu)
    frame_list = []
    for i, frame in enumerate(predictions):

        frame_grid = np.concatenate((source_image, driving_video[i], frame),axis=1)

        frame_list.append(img_as_ubyte(frame_grid))
    imageio.mimsave(opt.result_video, frame_list, fps=fps)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/ted384.yaml")
    parser.add_argument("--checkpoint", default='log/ted384_15_07_24_05.21.34/checkpoint_reconstruction.pth', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='/data/ysz/datasets/TEDTalk_dataset/00003/images/0001.png', help="path to source image")
    parser.add_argument("--driving_video", default='/data/ysz/datasets/TEDTalk_dataset/00003/images', help="path to driving video")
    parser.add_argument("--result_video", default='./result.mp4', help="path to output")
    parser.add_argument("--n_frames", default=150, type=int)
    parser.add_argument("--fps", default=25, type=int)

    parser.add_argument("--img_shape", default="384,384", type=lambda x: list(map(int, x.split(','))),
                        help='Shape of image, that the model was trained on.')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    main(parser.parse_args())
