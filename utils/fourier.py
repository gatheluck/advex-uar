import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import numpy as np
import torch
import torchvision

def fft(x:torch.Tensor, scale_factor:float=20.0):
    """
    Apply fft to the input tensor (RGB image).
    Args:
    - x (B,C,H,W): input tensor 
    - scale_factor: scale factor in frequency space
    Returns:
    - w (B,C,H,W):
    """
    assert len(x.size())==4
    assert x.size(1)==3
    x = x.detach().cpu()

    x_r = x[:,0,:,:]
    x_g = x[:,1,:,:]
    x_b = x[:,2,:,:]
    x_gray = (x_r + x_g + x_b)/3.0

    # apply fft
    w_r = _fft_np(x_r, scale_factor)
    w_g = _fft_np(x_g, scale_factor)
    w_b = _fft_np(x_b, scale_factor)
    w_gray = _fft_np(x_gray, scale_factor)
    out = torch.stack([w_r,w_g,w_b,w_gray], dim=1)

    del x, x_r, x_g, x_b
    del w_r, w_g, w_b
    return out

def _fft_np(x:torch.Tensor, scale_factor:float=20.0):
    """
    Apply np.fft.fft2 to the input tensor (gray image).
    Args:
    - x (B,H,W): input tensor
    - scale_factor: scale factor in frequency space
    Returns:
    - w (B,H,W): 
    """
    assert len(x.size())==3
    x = x.detach().cpu().numpy()
    
    w = np.fft.fft2(x) # https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft2.html#numpy.fft.fft2
    w = np.fft.fftshift(w, axes=(-2,-1)) # without specifying axes, this function also swap batch. 
    w = np.abs(w)+1 # add 1 to prevent -inf when apply log()
    w = scale_factor*np.log(w)
    w = w/np.max(w)
    out = torch.from_numpy(w).float()
    
    del x, w
    return out