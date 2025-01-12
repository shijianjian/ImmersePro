import os
import json
import argparse

import torch

import core
import core.inference


import cv2
from PIL import Image

import torchvision.transforms as transforms

from utils.file_client import FileClient
from utils.img_util import imfrombytes
from core.utils import Stack, ToTorchFormatTensor


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

        self.total_splits = args['total_splits']
        self.split_idx = args['split_idx']

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
        return sum(self.video_dict.values()) // self.num_local_frames // self.total_splits

    def __getitem__(self, index):
        selected_index = list(range(
            (index + self.split_idx * len(self)) * self.num_local_frames,
            (index + self.split_idx * len(self) + 1) * self.num_local_frames + self.num_ref_frames)
        )

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
        # normalizate, to tensors
        # frames_PIL = [np.array(f).astype(np.uint8) for f in frames]
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        frames_right_tensors = self._to_tensors(frames_right) * 2.0 - 1.0

        if self.load_flow:
            return frame_tensors, frames_right_tensors, flows_f, flows_b, video_name, frame_names
        else:
            return frame_tensors, frames_right_tensors, 'None', 'None', video_name, frame_names


def main_worker(config, dataloader=None, **kwargs):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    config['save_dir'] = os.path.join(
        config['save_dir'],
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    config['save_metric_dir'] = os.path.join(
        './scores',
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    config['device'] = torch.device("cuda:0")

    # trainer_version = config['trainer']['version']
    trainer = core.inference.Evaluator(config, **kwargs)
    # Trainer(config)
    trainer.inference(dataloader)


def main_worker_video_list(config, image_list, no_anaglyph=False, no_sbs=False, **kwargs):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    config['save_dir'] = os.path.join(
        config['save_dir'],
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    config['save_metric_dir'] = os.path.join(
        './scores',
        '{}_{}'.format(config['model']['net'],
                       os.path.basename(args.config).split('.')[0]))

    config['device'] = torch.device("cuda:0")

    # trainer_version = config['trainer']['version']
    trainer = core.inference.Evaluator(config, **kwargs)
    # Trainer(config)
    trainer.inference_video_list(image_list, no_anaglyph=no_anaglyph, no_sbs=no_sbs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='configs/train_propainter.json', type=str)
    parser.add_argument('--input_dir', default="./videos", type=str)
    parser.add_argument('--output_dir', default=".", type=str)
    parser.add_argument('--split_idx', default=0, type=int)
    parser.add_argument('--num_gpus', default=1, type=int)
    parser.add_argument('--no_anaglyph', action="store_true", default=False, help="If to output anaglyph version")
    parser.add_argument('--no_sbs', action="store_true", default=False, help="If to output sbs version")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    # loading configs
    config = json.load(open(args.config))

    # python inference_video.py  -c configs/train_deep3d_painter_v6_1_midas.json --output_dir /ibex/ai/home/shij0c/git/makeit3d/immersepro_clean
    base_dir = args.input_dir
    image_list = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith("_input.mp4")]
    main_worker_video_list(
        config, image_list=image_list, no_anaglyph=args.no_anaglyph, no_sbs=args.no_sbs, output_dir=args.output_dir, output_index=2
    )
