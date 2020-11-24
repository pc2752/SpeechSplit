# Generator functions to generate batches of data.

import numpy as np
import os
import time
import h5py
import random

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import soundfile as sf
import torchaudio
import utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() 

import matplotlib.pyplot as plt
import collections



class LoopDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, config, hparams, transform=None, output_size=128):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.mode = config.mode
        self.output_size = output_size
        self.transform = transform
        self.file_list = [os.path.join(x[0], y) for x in os.walk(os.path.join(self.root_dir,self.mode)) for y in x[2] if y.endswith('.wav')]
        self.max_len_pad = 192

        # [os.path.join(x[0], y) for x in os.walk('./ts_audios') for y in x[2] if y.endswith('.wav')]


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample, fs = sf.read(self.file_list[idx])

        if self.transform:
            sample = self.transform(sample)

        return sample


class STFT(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        self.stft_transform = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=1024, hop_length=512, normalized=True)

    def __call__(self, sample):

        return abs(utils.stft(sample))


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=128, padded_size=192):
        self.output_size = output_size
        self.padded_size = padded_size

    def __call__(self, sample):

        len_song = sample.shape[-1]
        new_len = self.output_size

        top = np.random.randint(0, len_song - new_len)

        sample = np.clip(sample[top: top + new_len, :].T, 0.0, 1.0)

        sample = torch.from_numpy(np.pad(sample, ((0,0),(0,self.padded_size-sample.shape[1])), 'constant', constant_values=-1e10))

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        return sample.type(torch.FloatTensor)


def get_dataset(root_dir, config, hparams):
	dataset = LoopDataset(root_dir, config, hparams, transform=transforms.Compose([STFT(), RandomCrop(hparams.max_len_seq, hparams.max_len_pad), ToTensor()]))
	return DataLoader(dataset, batch_size=16,shuffle=True, num_workers=5)