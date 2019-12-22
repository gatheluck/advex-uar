import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
import tqdm
import math

import torch
import torchvision

from advex_uar.common import FlagHolder
from advex_uar.common.pyt_common import get_attack

ATTACK_METHODS = [
    'pgd_linf', 
    'pgd_l2', 
    'fw_l1', 
    'jpeg_linf', 
    'jpeg_l2', 
    'jpeg_l1', 
    'elastic', 
    'fog', 
    'gabor', 
    'snow',
]

MODEL_NAMES = {
    'clean',
    # 'elastic-linf-16',
    # 'fog-linf-65536',
    # 'fw-l1-4.065689',
    # 'gabor-linf-3200',
    # 'jpeg-l1-131072',
    # 'jpeg-l2-256',
    # 'jpeg-linf-2',
    # 'pgd-l2-4800',
    # 'pgd-linf-32',
    # 'snow-linf-16',
} 

CLASSES = {
    'imagenet100': 100,
    'cifar10': 10,
}

weight_root = '/home/gatheluck/Data/adv-trained-models/models'
dataset_root = '/media/gatheluck/gathe-drive/datasets/ImageNet100/val'
log_root = '/home/gatheluck/Scratch/advex-uar/logs/attacks'

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

ATTACKS = {
    'pgd_linf'  : [8,        16,       32],
    'pgd_l2'    : [1200,     2400,     4800],
    'fw_l1'     : [1.016422, 2.032844, 4.065689], 
    'jpeg_linf' : [0.5,      1,        2], 
    'jpeg_l2'   : [64,       128,      256],
    'jpeg_l1'   : [16384,    32768,    65536],
    'elastic'   : [4,        8,        16],
    'fog'       : [256,      512,      1024],
    'gabor'     : [100,      200,      400],
    'snow'      : [0.25,     0.5,      1],
}

# make deterministic
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_step_size(epsilon, n_iters, use_max=False):
    if use_max:
        return epsilon
    else:
        return epsilon / math.sqrt(n_iters)

def unnormalize(x, mean, std):
    mean = torch.FloatTensor(mean).cuda()
    std  = torch.FloatTensor(std).cuda()
    return x.mul(std[None,:,None,None]).add(mean[None,:,None,None])

def apply_attacks(x:torch.Tensor, t:torch.Tensor, model, attacks=ATTACKS, dataset='imagenet100', 
                  n_iters=50, scale_each=True, scale_eps=True):
    """
    Apply all attacks to model.

    Args:
    - x (B,C,H,W): input tensor
    - t (B): target label
    - attacks:
        {method1 : [eps1, eps2, ...], 
         method2 : [eps3, eps4, ...], ...}
        
    Returns:
    - adv  (dict)
    - pert (dict):
        {method1 : x1 (B,num_eps,C,H,W),
         method2 : x2 (B,num_eps,C,H,W),...}  
    """
    assert len(x.size())==4
    assert x.size(1)==3

    model.eval()

    adv_dict  = dict()
    pert_dict = dict() 

    x = x.detach()
    x_unnormalized = unnormalize(x, MEAN, STD)

    for attack_method, eps_list in tqdm.tqdm(attacks.items()):
        if attack_method not in ATTACK_METHODS: raise ValueError

        print(attack_method)

        for eps in eps_list:
            step_size = get_step_size(eps, n_iters, use_max=False)
            
            attack = get_attack(dataset, attack_method, eps, n_iters, step_size, scale_each)
            attack = attack()
            
            print(attack.__class__)
            adv = attack(model, x, t, avoid_target=True, scale_eps=False).detach()

            # unnormalize
            adv_unnormalized = unnormalize(adv.detach(), MEAN, STD)
            delta = adv_unnormalized-x_unnormalized

            print('delta.min: ', torch.min(delta))
            print('delta.max: ', torch.max(delta))
            #pert = torch.clamp(adv-x, min=0.0, max=1.0)

            out = torch.cat([x_unnormalized, adv_unnormalized, delta], dim=-2)

            save_name = attack_method+'_{eps}.png'.format(eps=eps)
            save_path = os.path.join(log_root, save_name)

            torchvision.utils.save_image(out, save_path, nrow=16)

            del attack, adv, adv_unnormalized, delta, out 

@click.command()

@click.option("--batch_size", default=16)
@click.option("--num_workers", default=16)
@click.option("--scale_eps/--no_scale_eps", is_flag=True, default=True)

# def main(**flags):
#     FLAGS = FlagHolder()
#     FLAGS.initialize(**flags)

#     # dataset (ImageNet100)
#     normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                                 std=[0.229, 0.224, 0.225])
#     transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
#                                                 torchvision.transforms.CenterCrop(224),
#                                                 torchvision.transforms.ToTensor(),
#                                                 normalize,])

#     dataset = torchvision.datasets.ImageFolder(dataset_root, transform)
#     loader  = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers, shuffle=True)

#     for name in MODEL_NAMES:
#         # model
#         weight_path = os.path.join(weight_root, name+'_model.pth')
#         model = torchvision.models.resnet50(num_classes=100)
#         model.load_state_dict(torch.load(weight_path))
#         model.eval()
#         model = model.cuda()

#         for i, (x,t) in enumerate(loader):
#             x, t = x.cuda(), t.cuda()
#             apply_attacks(x, t, model)



#             break
        
def main(**flags):
    FLAGS = FlagHolder()
    FLAGS.initialize(**flags)

    # dataset (ImageNet100)
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                torchvision.transforms.CenterCrop(224),
                                                torchvision.transforms.ToTensor(),
                                                normalize,])

    dataset = torchvision.datasets.ImageFolder(dataset_root, transform)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers, shuffle=True)

    attacks = ATTACKS

    dataset='imagenet100'
    n_iters=50
    scale_each=True
    scale_eps=True

    for name in MODEL_NAMES:
        # model
        weight_path = os.path.join(weight_root, name+'_model.pth')
        model = torchvision.models.resnet50(num_classes=100)
        model.load_state_dict(torch.load(weight_path))
        model.eval()
        model = model.cuda()

        for i, (x,t) in enumerate(loader):
            x, t = x.cuda(), t.cuda()
            x_unnormalized = unnormalize(x.detach(), MEAN, STD)

            for attack_method, eps_list in tqdm.tqdm(attacks.items()):
                if attack_method not in ATTACK_METHODS: raise ValueError

                for eps in eps_list:
                    step_size = get_step_size(eps, n_iters, use_max=False)


                    attack = get_attack(dataset, attack_method, eps, n_iters, step_size, scale_each)
                    attack = attack()
                    
                    print(attack.__class__)
                    adv = attack(model, x, t, avoid_target=True, scale_eps=False).detach()

                    # unnormalize
                    adv_unnormalized = unnormalize(adv.detach(), MEAN, STD)
                    delta = adv_unnormalized-x_unnormalized

                    for idx in range(x.size(0)):

                        out = torch.cat([x_unnormalized[idx,:,:,:], 
                                           adv_unnormalized[idx,:,:,:], 
                                           delta[idx,:,:,:]], dim=-1).unsqueeze(0)

                        save_idx  = (FLAGS.batch_size*i)+idx
                        save_name = attack_method+'_{eps}_{save_idx:06d}.png'.format(eps=eps, save_idx=save_idx)
                        save_path = os.path.join(log_root, save_name)

                        torchvision.utils.save_image(out, save_path)

            if i==0: break

if __name__ == "__main__":
    main()