import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import torch
import torchvision

from PIL import Image

from utils.fourier import _fft_np, fft

TEST_IMAGE_PATH = os.path.join(base, 'tests','images','monkey.png')

def test_fft_np():
    x = Image.open(TEST_IMAGE_PATH)
    x = x.convert('L')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    x = transform(x)
    x = torch.cat([x, torch.randn(x.size()), torch.ones(x.size()), 0.3*torch.ones(x.size())+0.2*torch.randn(x.size())], dim=0)
    w = _fft_np(x)

    out_path = os.path.join(base, 'tests','out','fft_np.png')
    out = torch.cat([x.unsqueeze(1), w.unsqueeze(1)], dim=0)
    torchvision.utils.save_image(out, out_path, nrow=4)

def test_fft():
    x = Image.open(TEST_IMAGE_PATH)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    x = transform(x).unsqueeze(0)[:,0:3,:,:]
    r = torch.cat([torch.randn(1,1,x.size(-2),x.size(-1)),
                   torch.zeros(1,1,x.size(-2),x.size(-1)),
                   torch.zeros(1,1,x.size(-2),x.size(-1))], dim=1)
    g = torch.cat([torch.zeros(1,1,x.size(-2),x.size(-1)),
                   torch.randn(1,1,x.size(-2),x.size(-1)),
                   torch.zeros(1,1,x.size(-2),x.size(-1))], dim=1)
    b = torch.cat([torch.zeros(1,1,x.size(-2),x.size(-1)),
                   torch.zeros(1,1,x.size(-2),x.size(-1)),
                   torch.randn(1,1,x.size(-2),x.size(-1))], dim=1)
    x = torch.cat([x,
                   r, g, b,
                   torch.randn(x.size()), 
                   torch.ones(x.size()), 
                   0.3*torch.ones(x.size())+0.2*torch.randn(x.size())], dim=0)

    w = fft(x)
    w = w.view(-1,w.size(-2),w.size(-1)).unsqueeze(1)

    out_x_path = os.path.join(base, 'tests','out','fft_x.png')
    out_w_path = os.path.join(base, 'tests','out','fft_w.png')
    torchvision.utils.save_image(x, out_x_path)
    torchvision.utils.save_image(w, out_w_path, nrow=4)

if __name__ == '__main__':
    test_fft_np()
    test_fft()
