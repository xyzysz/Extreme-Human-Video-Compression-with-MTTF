
import sys
from tqdm import trange
import torch
from torch.utils.data import DataLoader
from logger import Logger
from modules.model import ReconstructionModel
from torch.optim.lr_scheduler import MultiStepLR
from sync_batchnorm import DataParallelWithCallback
from frames_dataset import DatasetRepeater
import random
import torch.nn.functional as F

def train(config, generator, region_predictor, mask_supervisor, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer = torch.optim.Adam(list(generator.parameters()) +
                                 list(region_predictor.parameters()), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, region_predictor, None, None,
                                      optimizer, None)
    else:
        start_epoch = 0

    scheduler = MultiStepLR(optimizer, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True,
                            num_workers=train_params['dataloader_workers'], drop_last=True)

    model = ReconstructionModel(region_predictor, generator, mask_supervisor, train_params)

    if torch.cuda.is_available():
        if ('use_sync_bn' in train_params) and train_params['use_sync_bn']:
            model = DataParallelWithCallback(model, device_ids=device_ids)
        else:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                uniform_number = random.uniform(0, 1)
                if uniform_number > 2/train_params['multiRes_mode']:
                    scale_factor = 0.25
                elif uniform_number > 1/train_params['multiRes_mode']:
                    scale_factor = 0.5
                else:
                    scale_factor = 1
                x['source'] = F.interpolate(x['source'], scale_factor=scale_factor)
                x['driving'] = F.interpolate(x['driving'], scale_factor=scale_factor)
        
                losses, generated = model(x, scale_factor)
                loss_values = [val.mean() for val in losses.values()]
                loss = sum(loss_values)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses.items()}
                logger.log_iter(losses=losses)

            scheduler.step()
            logger.log_epoch(epoch, {'generator': generator,
                                    'region_predictor': region_predictor,
                                    'optimizer_reconstruction': optimizer}, inp=x, out=generated)
            
       
