import os
import torch
import torch.utils.data as data
from utils import data_util

from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np

import cv2

# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif', 'bmp'])
#
# def train_val_list(enhan_images_path, ori_images_path):
#     image_list_index = os.listdir(ori_images_path)
#     all_length = len(image_list_index)
#     image_list_index = random.sample(image_list_index, all_length)
#
#     image_dataset = []
#     for i in image_list_index:  # Add paths and combine them
#         image_dataset.append((enhan_images_path + i, ori_images_path + i))
#
#     train_list = image_dataset[:int(all_length*0.9)]
#     val_list = image_dataset[int(all_length*0.9):]
#
#     return train_list, val_list
#
# class Train_Image_Dataset(data.Dataset):
#     def __init__(self, ori_images_path, enhan_images_path):
#         inp_files = sorted(os.listdir(ori_images_path))
#         tar_files = sorted(os.listdir(enhan_images_path))
#         self.inp_path = [os.path.join(ori_images_path, x) for x in inp_files if data_util.is_img_file(x)]
#         self.tar_apth = [os.path.join(enhan_images_path, x) for x in tar_files if data_util.is_img_file(x)]
#
#         print("Total training examples:", len(self.inp_path))
#
#     def __getitem__(self, index):
#
#         data_ori_path = self.inp_path[index]
#         data_clean_path = self.tar_apth[index]
#
#         data_clean = Image.open(data_clean_path)
#         data_ori = Image.open(data_ori_path)
#
#         data_clean = np.asarray(data_clean) / 255.0
#         data_ori = np.asarray(data_ori) / 255.0
#
#         data_clean = torch.from_numpy(data_clean).float()
#         data_ori = torch.from_numpy(data_ori).float()
#         return {
#             'raw': data_clean.permute(2, 0, 1),
#             'gt': data_ori.permute(2, 0, 1),
#         }
#
#     def __len__(self):
#         return len(self.inp_path)

class Train_Image_Dataset(data.Dataset):

    def __init__(self, raw_root, gt_root, random_transform=True):
        super(Train_Image_Dataset, self).__init__()
        assert os.path.exists(raw_root)
        assert os.path.exists(gt_root)
        self.raw_root = raw_root
        self.gt_root = gt_root
        self.random_transform = random_transform

        self.imgs_raw, self.imgs_gt = [], []

        # read data:
        self.imgs_raw = data_util.gen_file_list(self.raw_root)
        self.imgs_gt = data_util.gen_file_list(self.gt_root)

    def __getitem__(self, index):
        raw_path = self.imgs_raw[index]
        gt_path = self.imgs_gt[index]

        img_raw = data_util.read_img(raw_path)
        img_gt = data_util.read_img(gt_path)

        if self.random_transform:
            img_raw, img_gt = data_util.augment_img(img_raw, img_gt)
        return {
            'raws': img_raw,  # shape: [C, H, W]
            'gts': img_gt,  # shape: [C, H, W]
        }

    def __len__(self):
        return len(self.imgs_raw)

class Test_Image_Dataset(data.Dataset):

    def __init__(self, raw_root):
        super(Test_Image_Dataset, self).__init__()
        assert os.path.exists(raw_root)
        self.raw_root = raw_root

        self.imgs_raw = []
        # read data:
        self.imgs_raw = data_util.gen_file_list(self.raw_root)

    def __getitem__(self, index):
        raw_path = self.imgs_raw[index]
        img_raw = data_util.read_img(raw_path)
        name = self.imgs_raw[index].split('/')[-1]
        return {
            'raws': img_raw,  # shape: [C, N, H, W]
            'names': name
        }

    def __len__(self):
        return len(self.imgs_raw)

class Train_Video_Dataset(data.Dataset):

    def __init__(self, raw_root, gt_root, num_frames, random_transform=True):
        super(Train_Video_Dataset, self).__init__()
        assert os.path.exists(raw_root)
        assert os.path.exists(gt_root)
        self.raw_root = raw_root
        self.gt_root = gt_root
        self.num_frames = num_frames
        self.random_transform = random_transform

        self.folders = []
        self.imgs_raw, self.imgs_gt = [], []

        # read data:
        subfolders_raw = data_util.gen_file_list(self.raw_root)
        subfolders_gt = data_util.gen_file_list(self.gt_root)

        for subfolder_raw, subfolder_gt in zip(subfolders_raw, subfolders_gt):
            subfolder_name = os.path.basename(subfolder_raw)

            paths_raw = data_util.gen_file_list(subfolder_raw, self.num_frames)
            paths_gt = data_util.gen_file_list(subfolder_gt, self.num_frames)

            max_idx = len(paths_raw)
            assert max_idx == len(paths_gt), 'Different number of images in raw and gt folders'

            for i in range(max_idx):
                self.folders.append(subfolder_name)
            for sublist_raw, sublist_gt in zip(paths_raw, paths_gt):
                self.imgs_raw.append(sublist_raw)
                self.imgs_gt.append(sublist_gt)

    def __getitem__(self, index):
        raw_paths = self.imgs_raw[index]
        gt_paths = self.imgs_gt[index]

        imgs_raw, _ = data_util.read_img_seq(raw_paths)
        imgs_gt, _ = data_util.read_img_seq(gt_paths)

        raw_list = list(imgs_raw.unbind(0))
        gt_list = list(imgs_gt.unbind(0))

        if self.random_transform:
            raw_list, gt_list = data_util.random_augment(raw_list, gt_list)
        return {
            'raws': torch.stack(raw_list).permute(1, 0, 2, 3).contiguous(),  # shape: [C, N, H, W]
            'gts': torch.stack(gt_list).permute(1, 0, 2, 3).contiguous(),  # shape: [C, N, H, W]
        }

    def __len__(self):
        return len(self.folders)

class Test_Video_Dataset(data.Dataset):

    def __init__(self, raw_root, num_frames):
        super(Test_Video_Dataset, self).__init__()
        assert os.path.exists(raw_root)
        self.raw_root = raw_root
        self.num_frames = num_frames

        self.folders = []
        self.imgs_raw = []

        # read data:
        subfolders_raw = data_util.gen_file_list(self.raw_root)
        for subfolder_raw in subfolders_raw:
            subfolder_name = os.path.basename(subfolder_raw)

            paths_raw = data_util.gen_file_list(subfolder_raw, self.num_frames)

            for i in range(len(paths_raw)):
                self.folders.append(subfolder_name)
            for sublist_raw in paths_raw:
                self.imgs_raw.append(sublist_raw)

    def __getitem__(self, index):
        raw_paths = self.imgs_raw[index]
        imgs_raw, names = data_util.read_img_seq(raw_paths)
        raw_list = list(imgs_raw.unbind(0))
        return {
            'raws': torch.stack(raw_list).permute(1, 0, 2, 3).contiguous(),  # shape: [C, N, H, W]
            'names': names,
            'folder': self.folders[index],
        }

    def __len__(self):
        return len(self.folders)