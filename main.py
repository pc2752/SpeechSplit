import os
import argparse
import torch
from torch.backends import cudnn

from solver import Solver
# from data_loader import get_loader
from loop_dataset import get_dataset
from hparams import hparams, hparams_debug_string



def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)

    # Data loader.
    vcc_loader = get_dataset(config.root_dir, config, hparams)

    val_loader = get_dataset(config.val_root_dir, config, hparams)
    
    # Solver for training
    solver = Solver(vcc_loader, val_loader, config, hparams)

    solver.train()
    
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    # Training configuration.
    parser.add_argument('--root_dir', type=str, default='/home/pc2752/share/loop_synth/Balanced_Loops', help="Directory where the Balanced Loops are stored")
    parser.add_argument('--val_root_dir', type=str, default='/home/pc2752/share/loop_synth/Freesound_Loops', help="Directory where the Freesound Loops are stored")
    parser.add_argument('--mode', type=str, default='ts_audios', help="Mode to use, could be ts_audios or chopped audios")
    parser.add_argument('--num_iters', type=int, default=500000, help='number of total iterations')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')

    # Miscellaneous.
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--device_id', type=int, default=0)

    # Directories.
    parser.add_argument('--log_dir', type=str, default='run/logs')
    parser.add_argument('--sample_dir', type=str, default='run/samples')

    # Step size.
    parser.add_argument('--log_step', type=int, default=1000)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    print(hparams_debug_string())
    main(config)