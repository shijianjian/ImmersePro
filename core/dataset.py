import os
import json
import random

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import collections

from utils.file_client import FileClient
from utils.img_util import imfrombytes
from utils.flow_util import resize_flow, flowread
from core.utils import (create_random_shape_with_random_motion, Stack,
                        ToTorchFormatTensor, GroupRandomHorizontalFlip,GroupRandomHorizontalFlowFlip)
import kornia.augmentation as K


class TrainDataset_ProPainter(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args['video_root']
        self.flow_root = args['flow_root']
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.flow_root)

        json_path = os.path.join('./datasets', args['name'], 'train.json')

        with open(json_path, 'r') as f:
            self.video_train_dict = json.load(f)
        self.video_names = sorted(list(self.video_train_dict.keys()))

        # self.video_names = sorted(os.listdir(self.video_root))
        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list

        self.video_names = list(self.video_dict.keys()) # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            self.num_ref_frames)

        # read video frames
        frames = []
        masks = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)
            masks.append(all_masks[idx])

            if len(frames) <= self.num_local_frames-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, f'{next_n}_{current_n}_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if len(frames) == self.num_local_frames: # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    masks.reverse()
                    if self.load_flow:
                        flows_f.reverse()
                        flows_b.reverse()
                        flows_ = flows_f
                        flows_f = flows_b
                        flows_b = flows_

        if self.load_flow:
            frames, flows_f, flows_b = GroupRandomHorizontalFlowFlip()(frames, flows_f, flows_b)
        else:
            frames = GroupRandomHorizontalFlip()(frames)

        # normalizate, to tensors
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        # img [-1,1] mask [0,1]
        if self.load_flow:
            return frame_tensors, mask_tensors, flows_f, flows_b, video_name
        else:
            return frame_tensors, mask_tensors, 'None', 'None', video_name


class TestDataset_ProPainter(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = self.w, self.h = args['size']

        self.video_root = args['video_root']
        self.mask_root = args['mask_root']
        self.flow_root = args['flow_root']

        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.flow_root)
        self.video_names = sorted(os.listdir(self.mask_root))

        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            self.video_dict[v] = v_len
            self.frame_dict[v] = frame_list

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        selected_index = list(range(self.video_dict[video_name]))

        # read video frames
        frames = []
        masks = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            frame_path = os.path.join(self.video_root, video_name, frame_list[idx])

            img_bytes = self.file_client.get(frame_path, 'input')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)

            mask_path = os.path.join(self.mask_root, video_name, str(idx).zfill(5) + '.png')
            mask = Image.open(mask_path).resize(self.size, Image.NEAREST).convert('L')

            # origin: 0 indicates missing. now: 1 indicates missing
            mask = np.asarray(mask)
            m = np.array(mask > 0).astype(np.uint8)

            m = cv2.dilate(m,
                           cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                           iterations=4)
            mask = Image.fromarray(m * 255)
            masks.append(mask)

            if len(frames) <= len(selected_index)-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, f'{next_n}_{current_n}_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

        # normalizate, to tensors
        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)

        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        if self.load_flow:
            return frame_tensors, mask_tensors, flows_f, flows_b, video_name, frames_PIL
        else:
            return frame_tensors, mask_tensors, 'None', 'None', video_name


class TrainDataset(TrainDataset_ProPainter):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args['video_root']
        self.flow_root = args['flow_root']
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']
        self.size = self.w, self.h = (args['w'], args['h'])

        self.load_flow = args['load_flow']
        if self.load_flow:
            assert os.path.exists(self.flow_root)

        json_path = os.path.join('./datasets', args['name'], 'train.json')

        with open(json_path, 'r') as f:
            self.video_train_dict = json.load(f)
        self.video_names = sorted(list(self.video_train_dict.keys()))

        # self.video_names = sorted(os.listdir(self.video_root))
        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v, "left")))
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list

        self.video_names = list(self.video_dict.keys()) # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.augmentation = K.AugmentationSequential(
            K.Resize((self.h + int(self.h * 0.1), self.w + int(self.w * 0.1)),),
            K.RandomCrop((self.h, self.w), same_on_batch=True),
            same_on_batch=True
        )
        self.file_client = FileClient('disk')

    def __getitem__(self, index):
        video_name = self.video_names[index]

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            self.num_ref_frames)

        # read video frames
        frames = []
        frames_right = []
        flows_f, flows_b = [], []
        flows_ltr = []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, "left", frame_list[idx])
            right_img_path = os.path.join(self.video_root, video_name, "right", frame_list[idx].replace("left", "right"))
            img_bytes = self.file_client.get(img_path, 'img')
            right_img_bytes = self.file_client.get(right_img_path, 'right_img')

            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            right_img = imfrombytes(right_img_bytes, float32=False)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.resize(right_img, self.size, interpolation=cv2.INTER_LINEAR)
            right_img = Image.fromarray(right_img)

            frames.append(img)
            frames_right.append(right_img)

            # current_n = frame_list[idx][:-4]
            # flow_ltr_path = os.path.join(self.flow_root, video_name, "stereo", f'{current_n}_{current_n}_ltr.flo')
            # flow_ltr = flowread(flow_ltr_path, quantize=False)
            # flow_ltr = resize_flow(flow_ltr, self.h, self.w)
            # flows_ltr.append(flow_ltr)

            if len(frames) <= self.num_local_frames-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, "left", f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, "left", f'{next_n}_{current_n}_b.flo')
                # flow_f_path = os.path.join(self.flow_root, video_name, "left", f'frame_{str(current_n).zfill(8)}_left_frame_{str(next_n).zfill(8)}_left_f.flo')
                # flow_b_path = os.path.join(self.flow_root, video_name, "left", f'frame_{str(next_n).zfill(8)}_left_frame_{str(current_n).zfill(8)}_left_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if len(frames) == self.num_local_frames:  # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    frames_right.reverse()
                    if self.load_flow:
                        flows_f.reverse()
                        flows_b.reverse()
                        flows_ = flows_f
                        flows_f = flows_b
                        flows_b = flows_

        # if self.load_flow:
        #     frames, frames_right, flows_f, flows_b = GroupRandomHorizontalFlowFlip()(
        #         frames, frames_right, flows_f, flows_b)
        # else:
        #     frames, frames_right = GroupRandomHorizontalFlip()(frames, frames_right)

        # normalizate, to tensors
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        frames_right_tensors = self._to_tensors(frames_right) * 2.0 - 1.0
        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        # img [-1,1]
        if self.load_flow:
            frame_tensors, frames_right_tensors = self.augmentation(
                frame_tensors, frames_right_tensors,
                data_keys=["input", "input"]
            )
            flows_f, flows_b = self.augmentation(
                flows_f, flows_b,
                data_keys=["input", "input"]
            )
            return frame_tensors, frames_right_tensors, flows_f, flows_b, video_name
        else:
            frame_tensors, frames_right_tensors = self.augmentation(
                frame_tensors, frames_right_tensors,
                data_keys=["input", "input"]
            )
            return frame_tensors, frames_right_tensors, 'None', 'None', video_name


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.size = self.w, self.h = args['size']

        self.video_root = args['video_root']
        self.json_root = args['json_root']
        self.flow_root = args['flow_root']

        self.load_flow = args['load_flow']
        self.num_local_frames = args['num_local_frames']
        self.num_ref_frames = args['num_ref_frames']

        if self.load_flow:
            assert os.path.exists(self.flow_root)
        if self.json_root is not None:
            with open(self.json_root, 'r') as f:
                self.video_train_dict = json.load(f)
                self.video_names = sorted(list(self.video_train_dict.keys()))
        else:
            self.video_names = sorted(os.listdir(self.video_root))

        self.video_dict = {}
        self.frame_items = []

        for v in self.video_names:
            # frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v, "left")))
            v_len = len(frame_list)
            self.video_dict[v] = v_len
            self.frame_items += list(map(lambda x: (v, x), frame_list))

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return sum(self.video_dict.values()) // self.num_local_frames

    def __getitem__(self, index):
        selected_index = list(range(
            index * self.num_local_frames, (index + 1) * self.num_local_frames + self.num_ref_frames))

        # read video frames
        frames = []
        frames_right = []
        flows_f, flows_b = [], []
        video_name = None
        frame_names = []
        for idx in selected_index:
            _video_name, frame_name = self.frame_items[idx]
            if video_name is None:
                video_name = _video_name
            else:
                if video_name != _video_name:
                    continue
            frame_path = os.path.join(self.video_root, video_name, "left", frame_name)
            right_img_path = os.path.join(self.video_root, video_name, "right", frame_name.replace("left", "right"))
            # img_bytes = self.file_client.get(img_path, 'img')

            img_bytes = self.file_client.get(frame_path, 'input')
            right_img_bytes = self.file_client.get(right_img_path, 'output')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            right_img = imfrombytes(right_img_bytes, float32=False)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.resize(right_img, self.size, interpolation=cv2.INTER_LINEAR)
            right_img = Image.fromarray(right_img)

            frames.append(img)
            frames_right.append(right_img)
            frame_names.append(frame_name)

            # if len(frames) <= len(selected_index)-1 and self.load_flow:
            #     current_n = frame_list[idx][:-4]
            #     next_n = frame_list[idx+1][:-4]
            #     flow_f_path = os.path.join(self.flow_root, video_name, "left", f'{current_n}_{next_n}_f.flo')
            #     flow_b_path = os.path.join(self.flow_root, video_name, "left", f'{next_n}_{current_n}_b.flo')
            #     # flow_f_path = os.path.join(self.flow_root, video_name, "left", f'frame_{str(current_n).zfill(8)}_left_frame_{str(next_n).zfill(8)}_left_f.flo')
            #     # flow_b_path = os.path.join(self.flow_root, video_name, "left", f'frame_{str(next_n).zfill(8)}_left_frame_{str(current_n).zfill(8)}_left_b.flo')
            #     flow_f = flowread(flow_f_path, quantize=False)
            #     flow_b = flowread(flow_b_path, quantize=False)
            #     flow_f = resize_flow(flow_f, self.h, self.w)
            #     flow_b = resize_flow(flow_b, self.h, self.w)
            #     flows_f.append(flow_f)
            #     flows_b.append(flow_b)

        # normalizate, to tensors
        # frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        frames_right_tensors = self._to_tensors(frames_right) * 2.0 - 1.0

        # if self.load_flow:
        #     flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
        #     flows_b = np.stack(flows_b, axis=-1)
        #     flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
        #     flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        if self.load_flow:
            return frame_tensors, frames_right_tensors, flows_f, flows_b, video_name, frame_names
        else:
            return frame_tensors, frames_right_tensors, 'None', 'None', video_name, frame_names


def find_occlusions(flow_lr, flow_rl, threshold=.3):
    h, w, _ = flow_lr.shape
    occlusion_map = np.zeros((h, w), dtype=np.uint8)

    flow_lr = (flow_lr - flow_lr.min()) / (flow_lr.max() - flow_lr.min())
    flow_rl = (flow_rl - flow_rl.min()) / (flow_rl.max() - flow_rl.min())

    diff_flow = np.abs(flow_lr - flow_rl)
    # diff_flow = (np.abs(flow_lr - flow_rl) / abs(np.max(flow_lr, axis=-1, keepdims=True)))
    for y in range(h):
        for x in range(w):
            if np.any(diff_flow[y, x] > threshold):
                occlusion_map[y, x] = 255
    return occlusion_map


class TrainDatasetWithMask(TrainDataset):

    def __getitem__(self, index):
        video_name = self.video_names[index]

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            self.num_ref_frames)

        # read video frames
        frames = []
        frames_right = []
        masks = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, "left", frame_list[idx])
            right_img_path = os.path.join(self.video_root, video_name, "right", frame_list[idx].replace("left", "right"))
            img_bytes = self.file_client.get(img_path, 'img')
            right_img_bytes = self.file_client.get(right_img_path, 'right_img')

            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            right_img = imfrombytes(right_img_bytes, float32=False)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.resize(right_img, self.size, interpolation=cv2.INTER_LINEAR)
            right_img = Image.fromarray(right_img)

            frames.append(img)
            frames_right.append(right_img)

            if len(frames) <= self.num_local_frames-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, "left", f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, "left", f'{next_n}_{current_n}_b.flo')
                # flow_f_path = os.path.join(self.flow_root, video_name, "left", f'frame_{str(current_n).zfill(8)}_left_frame_{str(next_n).zfill(8)}_left_f.flo')
                # flow_b_path = os.path.join(self.flow_root, video_name, "left", f'frame_{str(next_n).zfill(8)}_left_frame_{str(current_n).zfill(8)}_left_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if self.load_flow:
                mask_path = flow_ltr_path = os.path.join(self.flow_root, video_name, "occ_mask_r", f'{current_n}.png')
                if not os.path.exists(mask_path):
                    flow_ltr_path = os.path.join(self.flow_root, video_name, "stereo", f'{current_n}_{current_n}_ltr.flo')
                    flow_rtl_path = os.path.join(self.flow_root, video_name, "stereo", f'{current_n}_{current_n}_rtl.flo')
                    flow_ltr = flowread(flow_ltr_path, quantize=False)
                    flow_rtl = flowread(flow_rtl_path, quantize=False)
                    flow_ltr = resize_flow(flow_ltr, self.h, self.w)
                    flow_rtl = resize_flow(flow_rtl, self.h, self.w)

                    mask = find_occlusions(flow_rtl, flow_ltr)
                    mask = Image.fromarray(mask)
                else:
                    mask = Image.open(mask_path).resize((self.w, self.h))
                masks.append(mask)

            if len(frames) == self.num_local_frames:  # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    frames_right.reverse()
                    if self.load_flow:
                        flows_f.reverse()
                        flows_b.reverse()
                        masks.reverse()
                        flows_ = flows_f
                        flows_f = flows_b
                        flows_b = flows_

        # if self.load_flow:
        #     frames, frames_right, flows_f, flows_b, masks = GroupRandomHorizontalFlowFlip()(
        #         frames, frames_right, flows_f, flows_b, masks)
        # else:
        #     frames, frames_right, masks = GroupRandomHorizontalFlip()(frames, frames_right, masks)

        # normalizate, to tensors
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        frames_right_tensors = self._to_tensors(frames_right) * 2.0 - 1.0
        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()
            masks = self._to_tensors(masks)

        # img [-1,1]
        if self.load_flow:
            return frame_tensors, frames_right_tensors, masks, flows_f, flows_b, video_name
        else:
            return frame_tensors, frames_right_tensors, masks, 'None', 'None', video_name


class TestDatasetWithMask(TestDataset):

    def __getitem__(self, index):
        video_name = self.video_names[index]
        selected_index = list(range(self.video_dict[video_name]))

        # read video frames
        frames = []
        masks = []
        frames_right = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            frame_path = os.path.join(self.video_root, video_name, "left", frame_list[idx])
            right_img_path = os.path.join(self.video_root, video_name, "right", frame_list[idx].replace("left", "right"))
            # img_bytes = self.file_client.get(img_path, 'img')

            img_bytes = self.file_client.get(frame_path, 'input')
            right_img_bytes = self.file_client.get(right_img_path, 'output')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            right_img = imfrombytes(right_img_bytes, float32=False)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.resize(right_img, self.size, interpolation=cv2.INTER_LINEAR)
            right_img = Image.fromarray(right_img)

            frames.append(img)
            frames_right.append(right_img)

            if len(frames) <= len(selected_index)-1 and self.load_flow:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, "left", f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, "left", f'{next_n}_{current_n}_b.flo')
                # flow_f_path = os.path.join(self.flow_root, video_name, "left", f'frame_{str(current_n).zfill(8)}_left_frame_{str(next_n).zfill(8)}_left_f.flo')
                # flow_b_path = os.path.join(self.flow_root, video_name, "left", f'frame_{str(next_n).zfill(8)}_left_frame_{str(current_n).zfill(8)}_left_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if self.load_flow:
                mask_path = flow_ltr_path = os.path.join(self.flow_root, video_name, "occ_mask_r", f'{current_n}.png')
                if not os.path.exists(mask_path):
                    flow_ltr_path = os.path.join(self.flow_root, video_name, "stereo", f'{current_n}_{current_n}_ltr.flo')
                    flow_rtl_path = os.path.join(self.flow_root, video_name, "stereo", f'{current_n}_{current_n}_rtl.flo')
                    flow_ltr = flowread(flow_ltr_path, quantize=False)
                    flow_rtl = flowread(flow_rtl_path, quantize=False)
                    flow_ltr = resize_flow(flow_ltr, self.h, self.w)
                    flow_rtl = resize_flow(flow_rtl, self.h, self.w)

                    mask = find_occlusions(flow_rtl, flow_ltr)
                    mask = Image.fromarray(mask)
                else:
                    mask = Image.open(mask_path).resize((self.w, self.h))
                masks.append(Image.fromarray(mask))

        # normalizate, to tensors
        frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        frames_right_tensors = self._to_tensors(frames_right) * 2.0 - 1.0

        if self.load_flow:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()
            masks = self._to_tensors(masks)

        if self.load_flow:
            return frame_tensors, frames_right_tensors, masks, flows_f, flows_b, video_name, frames_PIL
        else:
            return frame_tensors, frames_right_tensors, masks, 'None', 'None', video_name
