import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import random
import time
import numpy as np

from tqdm import tqdm
from thop import profile

from utils import config_util, model_util, data_util
from utils.dataloader import Test_Image_Dataset
from models.net_image import net_image

if __name__ == '__main__':
    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    opt = config_util.Config('configs/Image/test_config.yml', False)

    gpus = ','.join([str(i) for i in opt.TESTING.gpu_ids])
    print("gpu_id:", gpus)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.backends.cudnn.benchmark = True

    ######### Model ###########
    net_name = opt.MODEL.image_model_name
    model = net_image(opt.MODEL.n_feats)
    model.cuda()

    if len(gpus) > 1:
        print("Let's use", gpus, "GPUs!")
        model = nn.DataParallel(model, device_ids=opt.TESTING.gpu_ids)

    model_dir = os.path.join(opt.TESTING.checkpoint_path, net_name, 'model_best.pth')
    assert os.path.exists(model_dir)
    model_util.load_checkpoint(model, model_dir)
    print("===>Testing using weights: ", model_dir)

    ######### Dataset ###########
    test_raw_root = opt.DATASETS.test.raw_root
    dataset_name = opt.DATASETS.test.dataset_name
    test_dataset = Test_Image_Dataset(os.path.join(test_raw_root, dataset_name))
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                              drop_last=False, pin_memory=True)

    result_dir = os.path.join(opt.TESTING.result_path, net_name, dataset_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print('===> Loading datasets')

    with torch.no_grad():
        start_time = time.time()
        model.eval()
        for i, data in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            imgs_raw = data['raws'].cuda()
            file_names = data['names']

            imgs_out = model(imgs_raw)
            data_util.save_image(imgs_out[0][0], os.path.join(result_dir, file_names[0]))


    print("Time: {:.4f}".format(time.time() - start_time)+"\n")
    flops, params = profile(model, inputs=(imgs_raw,))
    print('flops(G): ', flops / 10 ** 9, 'params(M): ', params / 10 ** 6)
