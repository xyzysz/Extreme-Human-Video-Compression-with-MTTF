from torch import nn
import torch
import torch.nn.functional as F
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from torchvision import models
import numpy as np
from torch.autograd import grad
import SemanticGuidedHumanMatting
import torchvision.transforms as transforms
import random


class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict



class ReconstructionModel(torch.nn.Module):
    """
    Merge all updates into single model for better multi-gpu usage
    """

    def __init__(self, region_predictor, generator, mask_supervisor, train_params):
        super(ReconstructionModel, self).__init__()
        self.region_predictor = region_predictor
        self.generator = generator
        self.mask_supervisor = mask_supervisor
        self.train_params = train_params
        self.scales = train_params['scales']
        self.pyramid = ImagePyramide(self.scales, generator.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']
        self.multiRes_mode = generator.multiRes_mode
        if sum(self.loss_weights['perceptual']) != 0:
            self.vgg = Vgg19()
            if torch.cuda.is_available():
                self.vgg = self.vgg.cuda()
    
    def get_mask(self, model, img):
        infer_size = 1280
        h = img.size(2)
        w = img.size(3)
     
        if w >= h:
            rh = infer_size
            rw = int(w / h * infer_size)
        else:
            rw = infer_size
            rh = int(h / w * infer_size)
        rh = rh - rh % 64
        rw = rw - rw % 64    

        input_tensor = F.interpolate(img, size=(rh, rw), mode='bilinear')
        with torch.no_grad():
            pred = model(input_tensor)

        # progressive refine alpha
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
        pred_alpha = alpha_pred_os8.clone().detach()
        weight_os4 = SemanticGuidedHumanMatting.get_unknown_tensor_from_pred(pred_alpha, rand_width=30, train_mode=False)
        pred_alpha[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = SemanticGuidedHumanMatting.get_unknown_tensor_from_pred(pred_alpha, rand_width=15, train_mode=False)
        pred_alpha[weight_os1>0] = alpha_pred_os1[weight_os1>0]

        pred_alpha = pred_alpha.repeat(1, 3, 1, 1)
        pred_alpha = F.interpolate(pred_alpha, size=(h, w), mode='bilinear')
        
        return pred_alpha[:,0,:,:].unsqueeze(1)

    def forward(self, x,scale_factor):
        # print(x['source'].shape)
        driving_mask =  self.get_mask(self.mask_supervisor, x['driving']).detach()
        source_region_params = self.region_predictor(x['source'])
        driving_region_params = self.region_predictor(x['driving'])

        generated = self.generator(x['source'], scale_factor, source_region_params=source_region_params,
                                   driving_region_params=driving_region_params)
        prediction_mask = generated['mask_pred']

        generated['prediction'] = generated['prediction'] * prediction_mask + generated['bg'] * (1 - prediction_mask)
        
        generated.update({'source_region_params': source_region_params, 'driving_region_params': driving_region_params})

        loss_values = {}

        if self.loss_weights['bg'] != 0:
            value = torch.abs((generated['prediction'] - x['driving'])*(1-driving_mask)).mean()
            value += torch.abs(prediction_mask - driving_mask).mean()
            loss_values['bg'] = value* self.loss_weights['bg']

        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'])

        if sum(self.loss_weights['perceptual']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual'][i] * value
                loss_values['perceptual'] = value_total

        if self.loss_weights['pixel'] != 0:
            value = F.l1_loss(x['driving'], generated['prediction'])
            loss_values['pixel'] = self.loss_weights['pixel'] * value
    

        return loss_values, generated
