import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import random
import time
import numpy as np

from utils import config_util, model_util
from utils.dataloader import Train_Video_Dataset
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm

from models import loss
from models.net_image import net_image
from models.net_video import net_video
if __name__ == '__main__':
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    opt = config_util.Config('configs/Video/train_config.yml', True)

    gpus = ','.join([str(i) for i in opt.TRAINING.gpu_ids])
    print("gpu_id:", gpus)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.backends.cudnn.benchmark = True

    ######### Model ###########
    # image_model
    image_model_name = opt.MODEL.image_model_name
    model_image = net_image(opt.MODEL.n_feats)
    model_image.cuda()
    model_image_dir = os.path.join(opt.TRAINING.checkpoint_path, image_model_name, 'model_best.pth')
    assert os.path.exists(model_image_dir)
    model_util.load_checkpoint(model_image, model_image_dir)

    # video_model
    video_model_name = opt.MODEL.video_model_name
    video_model_dir = os.path.join(opt.TRAINING.checkpoint_path, video_model_name)
    if not os.path.exists(video_model_dir):
        os.makedirs(video_model_dir)
    num_frames = opt.TRAINING.num_frames
    model_video = net_video(opt.MODEL.n_feats, num_frames)
    model_video.cuda()

    if len(gpus) > 1:
        print("Let's use", gpus, "GPUs!")
        model_image = nn.DataParallel(model_image, device_ids=opt.TRAINING.gpu_ids)
        model_video = nn.DataParallel(model_video, device_ids=opt.TRAINING.gpu_ids)

    lr = opt.TRAINING.lr

    ######### Optimizer ###########
    optimizer = optim.Adam(model_video.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.TRAINING.num_epochs - warmup_epochs,
                                                            eta_min=lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Loss ###########
    criterion_char = loss.Charbonnier_Loss()
    criterion_mse = loss.MSE_Loss()
    # criterion_edge = loss.EdgeLoss()
    # criterion_vgg = loss.VggLoss()

    ######### Datasets ###########
    train_raw_root = opt.DATASETS.train.raw_root
    train_gt_root = opt.DATASETS.train.gt_root
    val_raw_root = opt.DATASETS.val.raw_root
    val_gt_root = opt.DATASETS.val.gt_root

    train_dataset = Train_Video_Dataset(train_raw_root, train_gt_root, num_frames, random_transform=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.TRAINING.batch_size, shuffle=True, num_workers=opt.TRAINING.num_workers,
                              drop_last=False, pin_memory=True)

    val_dataset = Train_Video_Dataset(val_raw_root, val_gt_root, num_frames, random_transform=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.TRAINING.batch_size, shuffle=False, num_workers=opt.TRAINING.num_workers,
                            drop_last=False, pin_memory=True)

    start_epoch = 1
    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.TRAINING.num_epochs + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(start_epoch, opt.TRAINING.num_epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = []
        psnr_list = []

        model_image.eval()
        model_video.train()
        for i, data in enumerate(tqdm(train_loader), 0):

            # zero_grad
            for param in model_video.parameters():
                param.grad = None

            #b, c, n, h, w
            imgs_raw = data['raws'].cuda()
            imgs_gt = data['gts'].cuda()
            best_img = imgs_raw[:, :, num_frames//2, :, :].clone()

            img_out = model_image(best_img)
            frames_en, _, _ = model_video(imgs_raw, img_out)

            # Compute loss at each stage
            loss_char = criterion_char(frames_en, imgs_gt)
            loss_mse = criterion_mse(frames_en, imgs_gt)
            loss_sum = loss_char + 2 * loss_mse

            loss_sum.backward()
            optimizer.step()
            epoch_loss.append(loss_sum.item())

        #### Evaluation ####
        model_image.eval()
        model_video.eval()
        for ii, data_val in enumerate((val_loader), 0):
            imgs_raw = data_val['raws'].cuda()
            imgs_gt = data_val['gts'].cuda()
            best_img = imgs_raw[:, :, num_frames//2, :, :].clone()

            with torch.no_grad():
                img_out = model_image(best_img)
                frames_en, _, _ = model_video(imgs_raw, img_out)
                psnr = loss.torchPSNR(frames_en, imgs_gt)
                psnr_list.append(psnr.item())

        psnr_val_rgb = np.mean(np.array(psnr_list))
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save({'state_dict': model_video.state_dict()}, os.path.join(video_model_dir, "model_best.pth"))

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))

        with open(os.path.join(video_model_dir, "val.log"), "a+", encoding="utf-8") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                    (epoch, psnr_val_rgb, best_epoch, best_psnr)+"\n")

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                  np.mean(epoch_loss), scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        with open(os.path.join(video_model_dir, "train.log"), "a+", encoding="utf-8") as f:
            f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                    np.mean(epoch_loss), scheduler.get_lr()[0])+"\n")