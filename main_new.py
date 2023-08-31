# from dataloader import dataset
from models import Twist_net
from utils import *
# from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
import random
import argparse

torch.manual_seed(0) # seed for cpu
torch.cuda.manual_seed(0) # seed for gpu
torch.cuda.manual_seed_all(0)  # if you are using multi-GPU, for all
np.random.seed(0)  # Numpy module.
random.seed(0)  # Python random module.	
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')
    
data_path = "/data/Mei/My_Reconstruction/icvl_28/Train/"
mask_path = "/data/Mei/My_Reconstruction/icvl_28/"
test_path = "/data/Mei/My_Reconstruction/icvl_28/Test/"

'''
data_path = "/data3/publicData/CASSI/Data/training/"
mask_path = "/data3/publicData/CASSI/Data/"
test_path = "/data3/publicData/CASSI/Data/testing/"
'''

# batch_size = 4
batch_size = 18
last_train = 0 # for finetune
model_save_filename = ' '  # for finetune
max_epoch = 5000
# learning_rate = 0.0001
learning_rate = 0.0001
epoch_sam_num = 1000
batch_num = int(np.floor(epoch_sam_num / batch_size))

mask3d_batch = generate_masks(mask_path)

train_set = LoadTraining(data_path)
test_data = LoadTest(test_path)

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device = torch.device('cuda:0')
model = Twist_net(nFrame=28, nPhase=5)
# model = nn.parallel.DistributedDataParallel(model)
# model = model.cuda()
# model = torch.nn.DataParallel(model)

torch.distributed.init_process_group(backend='nccl')
# model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)


if torch.cuda.device_count() > 1:
    model = nn.parallel.DistributedDataParallel(model.to(device))
    print("Let's use", torch.cuda.device_count(), "GPUs!")

if last_train != 0:
    model = torch.load('/data3/zhoushiyun/model/ /model_epoch_{}.pth'.format(last_train))
    # model = torch.load('./model/' + model_save_filename + '/model_epoch_{}.pth'.format(last_train))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
mse = torch.nn.MSELoss().cuda()


def train(epoch, learning_rate, logger):
    epoch_loss = 0
    begin = time.time()
    for i in range(batch_num):
        gt_batch = shuffle_crop(train_set, batch_size)
        gt = Variable(gt_batch).cuda().float()
        Xinput, Yinput = gen_meas_torch(gt, mask3d_batch)
        optimizer.zero_grad()
        Recon_img, Loss = model(Xinput, gt, mask3d_batch, mask3d_batch, Yinput)
        epoch_loss += Loss.data
        Loss.backward()
        optimizer.step()
    end = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".format(epoch, (epoch_loss / batch_num).item(),
                                                                                (end - begin)))


def test(epoch, logger):
    psnr_list, ssim_list = [], []
    processed_data = np.zeros((1, 28, 256, 256), dtype=np.float32)
    for i in range(10):
        frame_psnr_list, frame_ssim_list = [], []
        processed_data[0, :, :, :] = test_data[i][:, :, :]
        test_gt = torch.from_numpy(processed_data).cuda().float()
        # test_gt = test_data.cuda().float()
        test_PhiTy, Yinput = gen_meas_torch(test_gt, mask3d_batch)
        begin = time.time()
        with torch.no_grad():
            Recon_img, Loss = model(test_PhiTy, test_gt, mask3d_batch, mask3d_batch, Yinput)
            # weight_val = weightlist.detach().cpu().numpy()
        end = time.time()
        for k in range(test_gt.shape[1]):
            psnr_val = torch_psnr(Recon_img[:, k, :, :], test_gt[:, k, :, :])
            ssim_val = torch_ssim(Recon_img[:, k, :, :], test_gt[:, k, :, :])
            frame_psnr_list.append(psnr_val.detach().cpu().numpy())
            frame_ssim_list.append(ssim_val.detach().cpu().numpy())
            pred = np.transpose(Recon_img.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
            truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
            psnr_mean = np.mean(np.asarray(frame_psnr_list))
            ssim_mean = np.mean(np.asarray(frame_ssim_list))
            # weight_mean = np.mean(np.asarray(weight_list))
        psnr_list.append(psnr_mean)
        ssim_list.append(ssim_mean)
        # weight_all_list.append(weight_mean)
    all_psnr_mean = np.mean(np.asarray(psnr_list))
    all_ssim_mean = np.mean(np.asarray(ssim_list))
    # print('===> Epoch {}: frame{:}, testing psnr = {:.2f} time{:.2f}'.format(epoch, i, psnr_mean, (end - begin)))
    # print('PSNR'), print(np.asarray(psnr_list))
    # print('SSIM'), print(np.asarray(ssim_list))
    # psnr_list = []
    # ssim_list = []

    # print('===> Epoch {}: frame{:} testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(epoch, i, psnr_mean, ssim_mean, (end - begin)))
    # print('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(epoch, psnr_mean, ssim_mean, (end - begin)))

    logger.info(
        '===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(epoch, all_psnr_mean, all_ssim_mean, (end - begin)))
    model.train()
    return (pred, truth, psnr_list, ssim_list, all_psnr_mean, all_ssim_mean)


def checkpoint(epoch, model_path, logger):
    model_out_path = model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))


def main(learning_rate):
    if model_save_filename == '':
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
    else:
        date_time = model_save_filename
    result_path = '/data/Mei/recon/icvl_28/' + 'baseline_dy_cw' + date_time
    model_path = '/data/Mei/model/icvl_28/' + 'baseline_dy_cw' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(learning_rate, batch_size))
    psnr_max = 0

    for epoch in range(last_train + 1, last_train + max_epoch + 1):
        train(epoch, learning_rate, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)

        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 29:
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                # name = result_path + '/' + 'best.mat'
                scio.savemat(name, {'truth': truth, 'pred': pred, 'psnr_list': psnr_all, 'ssim_list': ssim_all})
                checkpoint(epoch, model_path, logger)

        # if (epoch % lr_epoch == 0) and (epoch < 200):
        # learning_rate = learning_rate * lr_scale
        # logger.info('Current learning rate: {}\n'.format(learning_rate))

        # if epoch % 10 == 0:
        # checkpoint(epoch, model_path, logger)


if __name__ == '__main__':
    main(learning_rate)


