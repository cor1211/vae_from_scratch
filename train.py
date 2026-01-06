import torch
from torch.utils.data import DataLoader
from model import VAE
from dataset import CartoonFacesDataset
from pathlib import Path
import yaml
from trainer import Trainer
import sys
from argparse import ArgumentParser
import random
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.optim import Adam

def load_yaml_config(config_path: str)-> dict:
    try:
        with open(Path(config_path), 'r') as file:
            config = yaml.safe_load(file)
            return config
    
    except FileNotFoundError:
        print('Config path is not valid')
        sys.exit(1)
    

def set_seed(seed:int = 42)-> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    parser = ArgumentParser(prog='Training Arguments')
    parser.add_argument('--config_path', type=str, required=True, help='The path to the yaml config file')
    args = parser.parse_args()

    #------------Load config file------------
    config_dict = load_yaml_config(args.config_path)
    cfg_data = config_dict['data']
    cfg_train = config_dict['train']

    #---------Set seed-------------
    seed = cfg_train.get('seed', 42)
    set_seed(seed)

    #---------Load Checkpoint-----------
    ckpt_path = cfg_train['resume_path']
    if ckpt_path:
        ckpt_path = Path(ckpt_path)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Resume_path: {ckpt_path} is not valid')
        try:
            run_name = ckpt_path.parent.name
        except:
            print(f'Cannot extract run_name from {ckpt_path}')
            sys.exit(1)
    else:
        run_name = f'exp_{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    
    #---------Init Sumary Writer to log------
    logdir = os.path.join('runs', run_name)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)
    print(f'TensorBoard logs will be saved in {logdir}')


    #------Dataset, Dataloader-----------
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5])
    ]
    ) 
        #----Dataset----
    train_set = CartoonFacesDataset(train=True, transform=transform, cfg_data=cfg_data)
    valid_set = CartoonFacesDataset(train=False, transform=transform, cfg_data=cfg_data)

        #-----Dataloader-------
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg_data['train_batch_size'], 
        shuffle=True,
        drop_last=True,
        num_workers=cfg_data['num_workers']
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg_data['valid_batch_size'],
        shuffle=False,
        drop_last=True,
        num_workers=cfg_data['num_workers']
    )

    #---------Configure model---------
    if not cfg_train['device']:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg_train['device'])
    print(f'Use device: {device}')

    model = VAE(in_channels=3, out_channels=3, dim_latent=100, device = device)
    optimizer = Adam(model.parameters(), lr = cfg_train['lr'], betas = tuple(cfg_train['betas']))


    #--------------Train--------------
    trainer = Trainer(model, optimizer, train_loader, valid_loader, device, config_dict, writer, run_name, ckpt_path)
    trainer.run()