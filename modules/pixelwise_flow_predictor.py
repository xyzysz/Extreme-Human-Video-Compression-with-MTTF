import sys
from torch import nn
import torch.nn.functional as F
import torch
from modules.util import Hourglass, make_coordinate_grid, region2gaussian
from modules.util import to_homogeneous, from_homogeneous, UpBlock2d, ResBlock2d


class PixelwiseFlowPredictor(nn.Module):
    """
    Module that predicts a pixelwise flow from sparse motion representation given by
    source_region_params and driving_region_params
    """

    def __init__(self, block_expansion, num_blocks, max_features, num_regions, num_feature, num_channels, num_background,
                 feature_size, estimate_occlusion_map=False,  use_deformed_source=True):
        super(PixelwiseFlowPredictor, self).__init__()
        self.hourglass_flow = Hourglass(block_expansion=block_expansion,
                                   in_features=num_feature * 2,
                                   max_features=max_features, num_blocks=num_blocks)
        self.flow = nn.Conv2d(self.hourglass_flow.out_filters, num_regions * 2, kernel_size=(7, 7), padding=(3, 3))

        self.hourglass = Hourglass(block_expansion=block_expansion,
                                   in_features=num_regions  * (num_channels * use_deformed_source + 1),
                                   max_features=max_features, num_blocks=num_blocks)

        self.bg_hourglass = Hourglass(block_expansion=block_expansion,
                                    in_features=6,
                                    max_features=max_features, num_blocks=num_blocks)

        self.mask = nn.Conv2d(self.hourglass.out_filters, num_regions, kernel_size=(7, 7), padding=(3, 3))

        if estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3))
        else:
            self.occlusion = None

        self.num_regions = num_regions
        self.num_background = num_background

        self.down = nn.UpsamplingBilinear2d(size=feature_size)
 
        self.use_deformed_source = use_deformed_source


    def create_heatmap_representations(self,source_image, driving_region_params, source_region_params):
        spatial_size = source_image.shape[2:]
        heatmap = torch.cat([driving_region_params, source_region_params], dim=1)

     
        heatmap = heatmap.unsqueeze(2)
        return heatmap


    def create_sparse_motions(self, source_image, flo, bg_params=None):
        bs, _, h, w = source_image.shape
        xx = torch.arange(0, w).view(1,-1).repeat(h,1).type(flo.type())
        yy = torch.arange(0, h).view(-1,1).repeat(1,w).type(flo.type())
        vgrid = torch.cat([xx.unsqueeze_(0), yy.unsqueeze_(0)], 0)
        # print('vgrid:', vgrid.size())

        vgrid = vgrid.view(1, 1, 2, h, w)
        
        #pixel flow motion
        vgrid = vgrid + flo 

        # scale grid to [-1,1] 
        vgrid[:,:,0,:,:] = 2.0*vgrid[:,:,0,:,:].clone()/max(w-1,1)-1.0 
        vgrid[:,:,1,:,:] = 2.0*vgrid[:,:,1,:,:].clone()/max(h-1,1)-1.0 # b, x, k, h, w
        vgrid = vgrid.permute(0,1,3,4,2)

        driving_to_source = vgrid

        sparse_motions = driving_to_source

        return sparse_motions

    def create_deformed_source_image(self, source_image, sparse_motions):
        bs, _, h, w = source_image.shape
        source_repeat = source_image.unsqueeze(1).unsqueeze(1).repeat(1, self.num_regions, 1, 1, 1, 1)
        source_repeat = source_repeat.view(bs * self.num_regions , -1, h, w)
        sparse_motions = sparse_motions.view((bs * self.num_regions , h, w, -1))
        sparse_deformed = F.grid_sample(source_repeat, sparse_motions)
        sparse_deformed = sparse_deformed.view((bs, self.num_regions , -1, h, w))
        return sparse_deformed

    def forward(self, source_image, driving_region_params, source_region_params):
        source_feature = source_region_params['feat']*source_region_params['coeff'] + source_region_params['bias']
        driving_feature = source_region_params['feat']*driving_region_params['coeff'] + driving_region_params['bias']
        
        source_image = self.down(source_image)

        bs, _, h, w = source_image.shape
        heatmap_representation = self.create_heatmap_representations(source_image, driving_feature, source_feature)
        out_dict = dict()
        flow_from_feature = self.hourglass_flow(torch.cat([source_feature, driving_feature],dim=1))  # b, k+1, 1, h/4, w/4
        flo = self.flow(flow_from_feature).reshape(bs, self.num_regions, 2, h, w)#.permute(0, 1, 3, 4, 2)
        sparse_motion = self.create_sparse_motions(source_image, flo) # b, k+1, h/4, w/4, 2
        
        deformed_source = self.create_deformed_source_image(source_image, sparse_motion) # b, k+1, 3, h/4, w/4        
        
        
        if self.use_deformed_source:
            predictor_input = torch.cat([heatmap_representation, deformed_source], dim=2)
        else:
            predictor_input = heatmap_representation
        predictor_input = predictor_input.view(bs, -1, h, w)

        prediction = self.hourglass(predictor_input) # b, 148, h/4, w/4
        

        mask = self.mask(prediction)
        mask, mask_bg = mask.split([self.num_regions-self.num_background,self.num_background],dim=1)
        mask = F.softmax(mask, dim=1) # b, k+1, h/4, w/4
        mask_bg = F.softmax(mask_bg, dim=1) # b, k+1, h/4, w/4
        
        mask = mask.unsqueeze(2)
        mask_bg = mask_bg.unsqueeze(2)
        sparse_motion, sparse_motion_bg = sparse_motion.split([self.num_regions-self.num_background,self.num_background],dim=1)
        sparse_motion = sparse_motion.permute(0, 1, 4, 2, 3)
        sparse_motion_bg = sparse_motion_bg.permute(0, 1, 4, 2, 3)
        deformation = (sparse_motion * mask).sum(dim=1)
        deformation_bg = (sparse_motion_bg * mask_bg).sum(dim=1)
        deformation = deformation.permute(0, 2, 3, 1)
        deformation_bg = deformation_bg.permute(0, 2, 3, 1)

        deformed_bg = F.grid_sample(source_image, deformation_bg) # b, k+1, 3, h/4, w/4

        bg_predictor_input = torch.cat([deformed_bg, source_image], dim=1)
        bg = self.bg_hourglass(bg_predictor_input)

        # bg = F.sigmoid(self.bg_up(bg))
        out_dict['bg'] = bg

        out_dict['optical_flow'] = deformation
        out_dict['optical_flow_bg'] = deformation_bg
        out_dict['deformed_source'] = deformed_source

        if self.occlusion:
            occlusion_map = torch.sigmoid(self.occlusion(prediction))
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
