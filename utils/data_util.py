import os
import cv2
import glob
import random
import numpy as np
from PIL import Image
from skimage import img_as_ubyte

import torch
import torchvision.transforms as trans
import torchvision.transforms.functional as TF

def is_img_file(filename):
    return any(filename.endswith(extension) for extension in
               ['jpg', 'jpeg', 'JPG', 'JPEG', 'png', 'PNG', 'gif', 'bmp', '.BMP'])

def gen_file_list(root_path, num_frame=None):
    if num_frame is None:
        return sorted(glob.glob(os.path.join(root_path, '*')))
    else:
        all_frames_path = sorted(glob.glob(os.path.join(root_path, '*')))
        single_frames_list = []
        all_frames_list = []
        for i in range(len(all_frames_path)):
            if (i+1) % num_frame == 0 and i != 0:
                single_frames_list.append(all_frames_path[i])
                all_frames_list.append(single_frames_list)
                single_frames_list = []
            else:
                single_frames_list.append(all_frames_path[i])
        return all_frames_list

def read_img(img_path):
    img = Image.open(img_path)
    img = np.asarray(img) / 255.0
    img = torch.from_numpy(img).float()
    return img.permute(2, 0, 1)

def read_img_seq(folder_path):
    """
    Read an image sequence from a folder path.
    Args:
        folder_path: Path to image sequence.
    Returns:
        img_seq: size (N, C, H, W), RGB, [0, 1].
        img_names: Returned image name list.
    """
    if isinstance(folder_path, list):
        img_paths = folder_path,
    else:
        img_paths = sorted(glob.glob(os.path.join(folder_path, '*')))
    imgs = [read_img(v) for v in img_paths[0]]
    img_seq = torch.stack(imgs, dim=0)
    img_names = [os.path.splitext(os.path.basename(path))[0] for path in img_paths[0]]
    return img_seq, img_names

def save_image(img_en, save_dir):
    img_en = torch.clamp(img_en, 0, 1)
    en_img = img_en[[2, 1, 0], :, :].permute(1, 2, 0).float().cpu().detach().numpy()
    en_img = img_as_ubyte(en_img)
    cv2.imwrite(save_dir, en_img)

def save_frames(imgs_en, result_dir, folder, file_names, mode):
    imgs_en = torch.clamp(imgs_en, 0, 1)
    for n in range(imgs_en.shape[1]):
        en_img = imgs_en[:, n, :, :].float().cpu().detach().numpy()
        if mode == 1:
            en_img = np.transpose(en_img[[2, 1, 0], :, :], (1, 2, 0)) # CHW to HWC, BGR to RGB for en_img,  bl
        if mode == 2:
            en_img = np.transpose(en_img, (1, 2, 0)) # CHW to HWC for t
        en_img = img_as_ubyte(en_img)
        cv2.imwrite((os.path.join(result_dir, folder[0], file_names[n][0] + '.png')), en_img)

def augment_img(img_raw, img_gt):
    # # Generate a random number to determine the augmentation type
    # aug = random.randint(0, 5)
    #
    # # Apply data augmentation based on the random number
    # if aug == 1:
    #     # Horizontal flip
    #     img_raw = TF.hflip(img_raw)
    #     img_gt = TF.hflip(img_gt)
    # elif aug == 2:
    #     # Vertical flip
    #     img_raw = TF.vflip(img_raw)
    #     img_gt = TF.vflip(img_gt)
    # elif aug == 3:
    #     # Random rotation
    #     degrees = random.uniform(-180, 180)
    #     img_raw = TF.rotate(img_raw, degrees)
    #     img_gt = TF.rotate(img_gt, degrees)
    # elif aug == 4:
    #     # Random color jitter
    #     color_transform = trans.ColorJitter(0.2, 0.2, 0.2, 0.2)
    #     # img_raw = color_transform(img_raw)
    #     # img_gt = color_transform(img_gt)
    # elif aug == 5:
    #     # Random crop
    #     h, w = img_raw.shape[1], img_raw.shape[2]
    #     h_ps = h // 2
    #     w_ps = w // 2
    #     h_c = random.randint(0, h - h_ps)
    #     w_c = random.randint(0, w - w_ps)
    #     # Crop the patch
    #     img_raw = img_raw[:, h_c:h_c + h_ps, w_c:w_c + w_ps]
    #     img_gt = img_gt[:, h_c:h_c + h_ps, w_c:w_c + w_ps]
    #     # Resize to original size
    #     resize_trans = trans.Resize((h, w))
    #     img_raw = resize_trans(img_raw)
    #     img_gt = resize_trans(img_gt)
    return img_raw, img_gt

def random_augment(raw_list, gt_list):
    """
    Randomly augment a list of raw images and their corresponding gt images.

    Args:
        raw_list (list): List of raw images.
        gt_list (list): List of gt images corresponding to raw images.

    Returns:
        tuple: Two lists, one for the augmented raw images and the other for the augmented ground truth images.
    """
    aug_raw_list, aug_gt_list = [], []

    # Iterate through each raw image and its corresponding ground truth image
    for img_raw, img_gt in zip(raw_list, gt_list):
        img_raw, img_gt = augment_img(img_raw, img_gt)

        # Append the augmented images to their respective lists
        aug_raw_list.append(img_raw)
        aug_gt_list.append(img_gt)

    # Return the lists of augmented images
    return aug_raw_list, aug_gt_list