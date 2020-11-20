from torch.utils.data import Dataset, DataLoader
from sve.pytorch.pytorch_generator import get_dataset
from sve.pytorch.pytorch_model import AutoVC
import os
import argparse
from torch.backends import cudnn
import torch


from sve.config import config as config_old





def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # print(torch.cuda.get_device_name(0))

    # # Data loader.
    
    dataloader = get_dataset(config, data="yam")

    valloader = get_dataset(config, data="nus")

    model = AutoVC(config,'yam_jeback')

    model.train(dataloader, valloader)
        
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_cd', type=float, default=1, help='weight for hidden code loss')
    parser.add_argument('--dim_neck', type=int, default=16)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=16)
    
    # Training configuration.
    parser.add_argument('--data_dir', type=str, default='./spmel')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=500000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=128, help='dataloader output sequence length')
    
    # Miscellaneous.
    parser.add_argument('--log_step', type=int, default=1000)
    parser.add_argument('--save_dir', type=str, default=config_old.log_dir)

    config = parser.parse_args()
    print(config)
    main(config)