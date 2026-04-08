import sys
from torch import nn
import torch
import torch.nn.functional as F
from modules.util import Hourglass, make_coordinate_grid, Encoder
from compressai.layers import GDN

class RegionPredictor(nn.Module):
    """
    Region estimating. Estimate affine parameters of the region.
    """

    def __init__(self, block_expansion, num_feature, num_channels, max_features,
                 num_blocks, feature_size, pad=3):
        super(RegionPredictor, self).__init__()

       
        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)

        self.regions = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_feature, kernel_size=(7, 7),
                                 padding=pad)

        self.coefficients = nn.Sequential(
            nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_feature, kernel_size=(5,5),stride=2, padding=2),
            GDN(num_feature), #48
            nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=(5,5),stride=2, padding=2),
            GDN(num_feature), #24
            nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=(5,5),stride=2, padding=2),
            GDN(num_feature), #12
            nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=(5,5),stride=2, padding=2),
            GDN(num_feature), #6
            nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=(5,5),stride=2, padding=2),
            GDN(num_feature), #3
            nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=(5,5),stride=3, padding=2),
        )

        self.bias = nn.Sequential(
            nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_feature, kernel_size=(5,5),stride=2, padding=2),
            GDN(num_feature), #48
            nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=(5,5),stride=2, padding=2),
            GDN(num_feature), #24
            nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=(5,5),stride=2, padding=2),
            GDN(num_feature), #12
            nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=(5,5),stride=2, padding=2),
            GDN(num_feature), #6
            nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=(5,5),stride=2, padding=2),
            GDN(num_feature), #3
            nn.Conv2d(in_channels=num_feature, out_channels=num_feature, kernel_size=(5,5),stride=3, padding=2),
        )
        self.down = nn.UpsamplingBilinear2d(size=feature_size)


    def forward(self, x):
 
        x = self.down(x) 
        feature_map = self.predictor(x) # b, 35, h/4, w/4
        prediction = self.regions(feature_map) 
        coefficients = self.coefficients(feature_map)
        bias = self.bias(feature_map)
        # print('coefficients:', coefficients.size())
        
        return {'feat': prediction, 'coeff': coefficients, 'bias' : bias}
       

