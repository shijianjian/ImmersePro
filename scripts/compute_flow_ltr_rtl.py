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


def find_occlusions(flow_lr, flow_rl):
    h, w, _ = flow_lr.shape
    occlusion_map = np.zeros((h, w), dtype=np.uint8)

    # Iterate over each pixel
    for y in range(h):
        for x in range(w):
            # Forward mapping coordinates
            dx, dy = flow_lr[y, x]
            x2 = int(x + dx)
            y2 = int(y + dy)

            # Check for boundaries
            if x2 < 0 or x2 >= w or y2 < 0 or y2 >= h:
                occlusion_map[y, x] = 255
                continue

            # Backward mapping coordinates
            dx2, dy2 = flow_rl[y2, x2]
            x_back = x2 + dx2
            y_back = y2 + dy2

            # Check for occlusion
            if abs(x_back - x) > 1 or abs(y_back - y) > 1:
                occlusion_map[y, x] = 255

    return occlusion_map

"""
srun --pty --time=24:00:00 --cpus-per-task=10 --gres=gpu:v100:1  bash -l
cd /ibex/ai/home/shij0c/git/makeit3d/ProPainter
conda activate makeit3d
python scripts/compute_flow.py --root_path /ibex/project/c2263/data/Youtube-SBS/extracted --save_path /ibex/project/c2263/data/Youtube-SBS/Flows_flo --height 280 --width 248
"""

if __name__ == '__main__':
    device = 'cuda'

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

    file_list = list([folder for folder in sorted(os.listdir(root_path)) if folder.startswith("yt_")])[args.starting_index:]

    # for i, f in enumerate(file_list):
    #     m_list = sorted(os.listdir(os.path.join(root_path, f, "left")))
    #     len_m = len(m_list)
    #     if os.path.exists(os.path.join(save_path, f, "stereo")):
    #         starting = max(0, len(os.listdir(os.path.join(save_path, f, "stereo"))) // 2 - 100)  # in case incompelete files saved
    #     else:
    #         starting = 0

    #     if os.path.exists(os.path.join(save_path, f, "stereo")) and len(os.listdir(os.path.join(save_path, f, "stereo"))) == len(os.listdir(os.path.join(root_path, f, "left"))) * 2:
    #         print(f'Skipping {i}/{len(file_list)}: {f} ...', flush=True)
    #         for i in range(starting, len_m):
    #             p = os.path.join(save_path, f, "stereo", f'{m_list[i][:-4]}_{m_list[i][:-4]}_ltr.flo')
    #             file_size = os.path.getsize(p)
    #             if 277772 != file_size:
    #                 assert False, (p, file_size)
    #         continue

    #     print(f'Processing {i}/{len(file_list)}: {f} ...', f"starting from file: {starting}/{len_m}", flush=True)
    #     for i in range(starting, len_m):
    #         img1_path = os.path.join(root_path, f, "left", m_list[i])
    #         img2_path = os.path.join(root_path, f, "right", m_list[i].replace("left", "right"))
    #         img1 = Image.fromarray(cv2.imread(img1_path))
    #         img2 = Image.fromarray(cv2.imread(img2_path))

    #         transform = transforms.Compose([transforms.ToTensor()])

    #         img1 = transform(img1).unsqueeze(0).to(device)[:,[2,1,0],:,:]
    #         img2 = transform(img2).unsqueeze(0).to(device)[:,[2,1,0],:,:]

    #         # upsize to a multiple of 16
    #         # h, w = img1.shape[2:4]
    #         # w_new = w if (w % 16) == 0 else 16 * (w // 16 + 1)
    #         # h_new = h if (h % 16) == 0 else 16 * (h // 16 + 1)


    #         img1 = F.interpolate(input=img1,
    #                             size=(h_new, w_new),
    #                             mode='bilinear',
    #                             align_corners=False)
    #         img2 = F.interpolate(input=img2,
    #                             size=(h_new, w_new),
    #                             mode='bilinear',
    #                             align_corners=False)

    #         with torch.no_grad():
    #           img1 = img1 * 2 - 1
    #           img2 = img2 * 2 - 1

    #           _, flow_ltr = RAFT_model(img1, img2, iters=20, test_mode=True)
    #           _, flow_rtl = RAFT_model(img2, img1, iters=20, test_mode=True)

    #         flow_ltr = flow_ltr[0].permute(1,2,0).cpu().numpy()
    #         flow_rtl = flow_rtl[0].permute(1,2,0).cpu().numpy()

    #         # flow_f = resize_flow(flow_f, w_new, h_new)
    #         # flow_b = resize_flow(flow_b, w_new, h_new)

    #         save_flow_ltr = os.path.join(save_path, f, "stereo", f'{m_list[i][:-4]}_{m_list[i][:-4]}_ltr.flo')
    #         save_flow_rtl = os.path.join(save_path, f, "stereo", f'{m_list[i][:-4]}_{m_list[i][:-4]}_rtl.flo')

    #         flowwrite(flow_ltr, save_flow_ltr, quantize=False)
    #         flowwrite(flow_rtl, save_flow_rtl, quantize=False)

    # Compute occlusion mask
    for i, f in enumerate(file_list):
        m_list = sorted(os.listdir(os.path.join(root_path, f, "left")))
        len_m = len(m_list)
        if os.path.exists(os.path.join(save_path, f, "occ_mask_r")):
            starting = max(0, len(os.listdir(os.path.join(save_path, f, "occ_mask_r"))) // 2 - 100)  # in case incompelete files saved
        else:
            starting = 0

        if os.path.exists(os.path.join(save_path, f, "occ_mask_r")) and len(os.listdir(os.path.join(save_path, f, "occ_mask_r"))) == len(os.listdir(os.path.join(root_path, f, "left"))):
            print(f'Skipping {i}/{len(file_list)}: {f} ...', flush=True)
            continue

        print(f'Processing {i}/{len(file_list)}: {f} ...', f"starting from file: {starting}/{len_m}", flush=True)
        os.makedirs(os.path.join(save_path, f, "occ_mask_r"), exist_ok=True)
        for i in range(starting, len_m):
            flow_ltr_path = os.path.join(save_path, f, "stereo", f'{m_list[i][:-4]}_{m_list[i][:-4]}_ltr.flo')
            flow_rtl_path = os.path.join(save_path, f, "stereo", f'{m_list[i][:-4]}_{m_list[i][:-4]}_rtl.flo')

            flow_ltr = flowread(flow_ltr_path, quantize=False)
            flow_rtl = flowread(flow_rtl_path, quantize=False)
            output_path = os.path.join(save_path, f, "occ_mask_r", f'{m_list[i][:-4]}.png')
            mask = find_occlusions(flow_rtl, flow_ltr)
            Image.fromarray(mask).save(output_path)
