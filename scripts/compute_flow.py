# -*- coding: utf-8 -*-
import sys 
sys.path.append(".") 

import os
import cv2
import argparse
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from RAFT import RAFT
from utils.flow_util import *

def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)

def initialize_RAFT(model_path='weights/raft-things.pth', device='cuda'):
    """Initializes the RAFT model.
    """
    args = argparse.ArgumentParser()
    args.raft_model = model_path
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.raft_model))

    model = model.module
    model.to(device)
    model.eval()

    return model

"""
srun --pty --time=24:00:00 --cpus-per-task=10 --gres=gpu:v100:1  bash -l
cd /ibex/ai/home/shij0c/git/makeit3d/ProPainter
conda activate makeit3d
python scripts/compute_flow.py --root_path /ibex/project/c2263/data/Youtube-SBS/extracted --save_path /ibex/project/c2263/data/Youtube-SBS/Flows_flo --height 280 --width 248
python scripts/compute_flow.py --root_path /ibex/ai/project/c2263/data/KITTI/images/training --save_path /ibex/ai/project/c2263/data/KITTI/images/training_flo --height 336 --width 1216
"""

if __name__ == '__main__':
    device = 'cuda'
    side = "left"

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--root_path', type=str, default='your_dataset_root/youtube-vos/JPEGImages')
    parser.add_argument('-o', '--save_path', type=str, default='your_dataset_root/youtube-vos/Flows_flo')
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--width', type=int, default=432)
    parser.add_argument('--starting_index', type=int, default=0)

    args = parser.parse_args()
    
    # Flow model
    RAFT_model = initialize_RAFT(device=device)  
    
    root_path = args.root_path
    save_path = args.save_path
    h_new, w_new = (args.height, args.width)
    
    # file_list = list([folder for folder in sorted(os.listdir(root_path)) if folder.startswith("yt_")])[args.starting_index:]
    file_list = list([folder for folder in sorted(os.listdir(root_path))])[args.starting_index:]
    print("started", len(file_list), file_list)
    for i, f in enumerate(file_list):
        m_list = sorted(os.listdir(os.path.join(root_path, f, side)))
        len_m = len(m_list)

        if os.path.exists(os.path.join(save_path, f, side)) and len(os.listdir(os.path.join(save_path, f, side))) == len(os.listdir(os.path.join(root_path, f, side))) * 2 - 2:
            print(f'Skipping {i}/{len(file_list)}: {f} ...')
            continue

        if os.path.exists(os.path.join(save_path, f, side)):
            starting = max(0, len(os.listdir(os.path.join(save_path, f, side))) // 2 - 100)  # in case incompelete files saved
        else:
            starting = 0

        print(f'Processing {i}/{len(file_list)}: {f} ...', f"starting from file: {starting}/{len_m}")
        for i in range(starting, len_m-1):
            img1_path = os.path.join(root_path, f, side, m_list[i])
            img2_path = os.path.join(root_path, f, side, m_list[i+1])
            img1 = Image.fromarray(cv2.imread(img1_path))
            img2 = Image.fromarray(cv2.imread(img2_path))

            transform = transforms.Compose([transforms.ToTensor()])

            img1 = transform(img1).unsqueeze(0).to(device)[:,[2,1,0],:,:]
            img2 = transform(img2).unsqueeze(0).to(device)[:,[2,1,0],:,:]

            # upsize to a multiple of 16
            # h, w = img1.shape[2:4]
            # w_new = w if (w % 16) == 0 else 16 * (w // 16 + 1)
            # h_new = h if (h % 16) == 0 else 16 * (h // 16 + 1)


            img1 = F.interpolate(input=img1,
                                size=(h_new, w_new),
                                mode='bilinear',
                                align_corners=False)
            img2 = F.interpolate(input=img2,
                                size=(h_new, w_new),
                                mode='bilinear',
                                align_corners=False)

            with torch.no_grad():
              img1 = img1*2 - 1
              img2 = img2*2 - 1

              _, flow_f = RAFT_model(img1, img2, iters=20, test_mode=True)
              _, flow_b = RAFT_model(img2, img1, iters=20, test_mode=True)


            flow_f = flow_f[0].permute(1,2,0).cpu().numpy()
            flow_b = flow_b[0].permute(1,2,0).cpu().numpy()

            # flow_f = resize_flow(flow_f, w_new, h_new)
            # flow_b = resize_flow(flow_b, w_new, h_new)

            save_flow_f = os.path.join(save_path, f, side, f'{m_list[i][:-4]}_{m_list[i+1][:-4]}_f.flo')
            save_flow_b = os.path.join(save_path, f, side, f'{m_list[i+1][:-4]}_{m_list[i][:-4]}_b.flo')
            
            flowwrite(flow_f, save_flow_f, quantize=False)
            flowwrite(flow_b, save_flow_b, quantize=False)
