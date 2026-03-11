import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import psutil
import random
import time
import numpy as np

from tqdm import tqdm
from thop import profile

from utils import config_util, model_util, data_util
from utils.dataloader import Test_Video_Dataset
from models.net_image import net_image
from models.net_video import net_video

if __name__ == '__main__':
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    opt = config_util.Config('configs/Video/test_config.yml', False)

    gpus = ','.join([str(i) for i in opt.TESTING.gpu_ids])
    print("gpu_id:", gpus)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.backends.cudnn.benchmark = True

    ######### Model ###########
    # image_model
    image_model_name = opt.MODEL.image_model_name
    model_image = net_image(opt.MODEL.n_feats)
    model_image.cuda()
    model_image_dir = os.path.join(opt.TESTING.checkpoint_path, image_model_name, 'model_best.pth')
    assert os.path.exists(model_image_dir)
    model_util.load_checkpoint(model_image, model_image_dir)
    print("===>Testing using image model weights: ", model_image_dir)

    # video_model
    video_net_name = opt.MODEL.video_model_name
    num_frames = opt.TESTING.num_frames
    model_video = net_video(opt.MODEL.n_feats, num_frames)
    model_video.cuda()
    model_video_dir = os.path.join(opt.TESTING.checkpoint_path, video_net_name, 'model_best.pth')
    assert os.path.exists(model_video_dir)
    model_util.load_checkpoint(model_video, model_video_dir)
    print("===>Testing using video model weights: ", model_video_dir)

    if len(gpus) > 1:
        print("Let's use", gpus, "GPUs!")
        model_image = nn.DataParallel(model_image, device_ids=opt.TESTING.gpu_ids)
        model_video = nn.DataParallel(model_video, device_ids=opt.TESTING.gpu_ids)

    ######### Dataset ###########
    test_raw_root = opt.DATASETS.test.raw_root
    dataset_name = opt.DATASETS.test.dataset_name

    test_dataset = Test_Video_Dataset(os.path.join(test_raw_root, dataset_name), num_frames)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8,
                              drop_last=False, pin_memory=True)

    result_dir = os.path.join(opt.TESTING.result_path, video_net_name, dataset_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print('===> Loading datasets')

    with torch.no_grad():
        start_time = time.time()
        model_image.eval()
        model_video.eval()
        for i, data in enumerate(tqdm(test_loader), 0):
            imgs_raw = data['raws'].cuda()
            file_names = data['names']
            floder = data['folder']
            best_img = imgs_raw[:, :, num_frames//2, :, :]
            if not os.path.exists((os.path.join(result_dir, floder[0]))):
                os.mkdir(os.path.join(result_dir, floder[0]))

            img_out = model_image(best_img)
            imgs_en, bl, t = model_video(imgs_raw, img_out)

            data_util.save_frames(imgs_en[0], result_dir, floder, file_names, 1)

    print("All Time: {:.4f}".format(time.time() - start_time))
    print("Single Frame Latency: {:.4f}".format((time.time() - start_time)/(len(test_loader)*num_frames)) + "\n")
    flops1, params1 = profile(model_image, inputs=(best_img, ))
    flops2, params2 = profile(model_video, inputs=(imgs_raw, img_out))
    print('flops(G): ', (flops1 + flops2) / 10 ** 9, 'params(M): ', (params1 + params2) / 10 ** 6)