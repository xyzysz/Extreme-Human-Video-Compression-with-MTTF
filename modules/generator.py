import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from modules.pixelwise_flow_predictor import PixelwiseFlowPredictor
from torchvision.utils import save_image
import numpy as np

class Generator(nn.Module):
    """
    Generator that given source image and region parameters try to transform image according to movement trajectories
    induced by region parameters. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_regions, num_feature, feature_size, num_background, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, multiRes_mode, pixelwise_flow_predictor_params=None, skips=False):
        super(Generator, self).__init__()
        assert multiRes_mode <= num_down_blocks, 'Current multiRes_mode is not compatible with feauture/input resolution setting'
        if pixelwise_flow_predictor_params is not None:
            self.pixelwise_flow_predictor = PixelwiseFlowPredictor(num_regions=num_regions, num_feature=num_feature, num_channels=num_channels,
                                                                   num_background=num_background, feature_size=feature_size, **pixelwise_flow_predictor_params)
        else:
            self.pixelwise_flow_predictor = None
        #================= first_blocks =================#
        first_blocks = []
        for i in range(multiRes_mode):
            first_blocks.append(SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3)))
        self.first_blocks = nn.ModuleList(first_blocks)
        #================= Down =================#
        for stage in range(num_down_blocks):
            down_blocks = []
            in_features = min(max_features, block_expansion * (2 ** stage))
            out_features = min(max_features, block_expansion * (2 ** (stage + 1)))
            for i in range(max(multiRes_mode-stage,1)):
                if i == 0:
                    down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
                else:
                    down_blocks.append(SameBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            setattr(self, 'down_blocks'+str(stage+1), nn.ModuleList(down_blocks))
        #================= bottleneck_blocks =================#
        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))
        #================= Up =================#
        for stage in range(num_down_blocks):
            up_blocks = []
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks-stage)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks-stage-1)))
            for i in range(min(stage+1,multiRes_mode)):
                if i == 0:
                    up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
                else:
                    up_blocks.append(SameBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            setattr(self, 'up_blocks'+str(stage+1), nn.ModuleList(up_blocks))
        #================= final_blocks =================#
        final_blocks = []
        for i in range(multiRes_mode):
            final_blocks.append(nn.Conv2d(block_expansion, num_channels+1, kernel_size=(7, 7), padding=(3, 3)))
        self.final_blocks = nn.ModuleList(final_blocks)
        #================= bg =================#
        self.bg_first = nn.Conv2d(self.pixelwise_flow_predictor.bg_hourglass.out_filters, block_expansion, kernel_size=(7, 7), padding=(3, 3))
        for stage in range(num_down_blocks):
            bg_blocks = []
            for i in range(min(stage+1,multiRes_mode)):
                if i == 0:
                    bg_blocks.append(UpBlock2d(block_expansion, block_expansion, kernel_size=(3, 3), padding=(1, 1)))
                    bg_blocks.append(ResBlock2d(block_expansion, kernel_size=(3, 3), padding=(1, 1)))
                else:
                    bg_blocks.append(ResBlock2d(block_expansion, kernel_size=(3, 3), padding=(1, 1)))
                    bg_blocks.append(ResBlock2d(block_expansion, kernel_size=(3, 3), padding=(1, 1)))
            setattr(self, 'bg_blocks'+str(stage+1), nn.ModuleList(bg_blocks))
        self.bg_final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        #================= attributes =================#
        self.num_channels = num_channels
        self.multiRes_mode = multiRes_mode
        self.num_down_blocks = num_down_blocks

    @staticmethod
    def deform_input(inp, optical_flow):
        _, h_old, w_old, _ = optical_flow.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            optical_flow = optical_flow.permute(0, 3, 1, 2)
            optical_flow = F.interpolate(optical_flow, size=(h, w), mode='bilinear')
            optical_flow = optical_flow.permute(0, 2, 3, 1)
        return F.grid_sample(inp, optical_flow)

    def apply_optical(self, input_previous=None, input_skip=None, motion_params=None):
        if motion_params is not None:
            if 'occlusion_map' in motion_params:
                occlusion_map = motion_params['occlusion_map']
            else:
                occlusion_map = None
            deformation = motion_params['optical_flow']
            input_skip = self.deform_input(input_skip, deformation)

            if occlusion_map is not None:
                if input_skip.shape[2] != occlusion_map.shape[2] or input_skip.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=input_skip.shape[2:], mode='bilinear')
                if input_previous is not None:
                    input_skip = input_skip * occlusion_map + input_previous * (1 - occlusion_map)
                else:
                    input_skip = input_skip * occlusion_map
            out = input_skip
        else:
            out = input_previous if input_previous is not None else input_skip
        return out

    def forward(self, source_image, scale_factor ,driving_region_params, source_region_params):
        
        module_index = int(np.log2(int(1/scale_factor)))
        out = self.first_blocks[module_index](source_image)
        skips = [out]
        for i in range(self.num_down_blocks):
            out = getattr(self, 'down_blocks'+str(i+1))[max(module_index-i,0)](out)
            skips.append(out)
        output_dict = {}
        if self.pixelwise_flow_predictor is not None:
            motion_params = self.pixelwise_flow_predictor(source_image=source_image,
                                                          driving_region_params=driving_region_params,
                                                          source_region_params=source_region_params)
            output_dict["deformed"] = self.deform_input(source_image, motion_params['optical_flow'])
            if 'occlusion_map' in motion_params:
                output_dict['occlusion_map'] = motion_params['occlusion_map']
        else:
            motion_params = None

        out = self.apply_optical(input_previous=None, input_skip=out, motion_params=motion_params)

        out = self.bottleneck(out)
        for i in range(self.num_down_blocks):
            out = self.apply_optical(input_skip=skips[-(i + 1)], input_previous=out, motion_params=motion_params)
            out = getattr(self, 'up_blocks'+str(i+1))[max(module_index-self.num_down_blocks+1+i,0)](out)
        out = self.apply_optical(input_skip=skips[0], input_previous=out, motion_params=motion_params)
        out = self.final_blocks[module_index](out)
        out, mask_pred = out.split([self.num_channels, 1], dim=1)
        out = F.sigmoid(out)
        output_dict['mask_pred'] = F.sigmoid(mask_pred)
       
        out = self.apply_optical(input_skip=source_image, input_previous=out, motion_params=motion_params)

        bg = self.bg_first(motion_params['bg'])
        for i in range(self.num_down_blocks):
            bg = getattr(self, 'bg_blocks'+str(i+1))[max(module_index-self.num_down_blocks+1+i,0)](bg)
        bg = self.bg_final(bg)
        bg = F.sigmoid(bg)
        
        output_dict['prediction'] = out
        output_dict['deformed_source'] = motion_params['deformed_source']
        output_dict['optical_flow'] = motion_params['optical_flow']
        output_dict['optical_flow_bg'] = motion_params['optical_flow_bg']
        output_dict['bg'] = bg

        return output_dict
