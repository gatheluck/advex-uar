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

@click.option('-t', '--trg_root', default='/home/gatheluck/Scratch/advex-uar/logs/attacks')
@click.option('-l', '--log_root', default='/home/gatheluck/Scratch/advex-uar/logs/fourier_analysis')

def main(**kwargs):

    flags = FlagHolder()
    flags.initialize(**kwargs)
    flags.summary()

    # make logdir
    os.makedirs(flags.log_root, exist_ok=True)

    trg_path = os.path.join(flags.trg_root, '*.png')
    path_list = sorted(glob.glob(trg_path))

    #print(path_list)

    for png_path in path_list:
        
        img = Image.open(png_path)

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        # load delta
        x = transform(img).unsqueeze(0) # (B, RGB, H, 3*W)
        width = int(x.size(-1)/3)
        delta = x[:,:,:,width*2:]

        # fft
        w = fft(delta) #(B,4,H,W)
        w = torch.cat([w[:,0,:,:], w[:,1,:,:], w[:,2,:,:], w[:,3,:,:]], dim=-1)
        w = w.unsqueeze(0).repeat(1,3,1,1) # (B, RGB, H, 4*W)

        out = torch.cat([x,w], dim=-1)

        # save
        basename = os.path.basename(png_path)
        name, ext = os.path.splitext(basename)
        savename = name + '_fft' + ext
        savepath = os.path.join(flags.log_root, savename)

        torchvision.utils.save_image(out, savepath)

if __name__ == '__main__':
    main()