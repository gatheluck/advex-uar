import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
import glob
import tqdm

import torch
import torchvision
from PIL import Image

from dada.fourier import fft
from advex_uar.common import FlagHolder

@click.command()

@click.option('-t', '--trg_root', default='/home/gatheluck/Scratch/advex-uar/logs/fft')
@click.option('-l', '--log_root', default='/home/gatheluck/Scratch/advex-uar/logs/fft_merge')
@click.option('-n', '--num_image', default=16)

def main(**kwargs):

    flags = FlagHolder()
    flags.initialize(**kwargs)
    flags.summary()

    # make logdir
    os.makedirs(flags.log_root, exist_ok=True)

    for idx in tqdm.tqdm(range(flags.num_image)):

        trgname = '*_{idx:06d}_fft.png'.format(idx=idx)
        trgpath = os.path.join(flags.trg_root, trgname)
    
        pngpaths = sorted(glob.glob(trgpath))
        outs = []

        if not pngpaths:
            print('No target files are found.')
            continue

        for pngpath in pngpaths:
            img = Image.open(pngpath)
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            x = transform(img).unsqueeze(0) # (B, RGB, H, 7*W)
            outs.append(x)

        out = torch.cat(outs, dim=-2) 

        # save  
        savename = '{idx:06d}_merge.png'.format(idx)
        savepath = os.path.join(flags.log_root, savename)
        torchvision.utils.save_image(out, savepath)
        raise NotImplementedError

if __name__ == '__main__':
    main()