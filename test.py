#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
from ptflops import get_model_complexity_info

from networks import ENet

BATCHSIZE_PER_CARD = 4

class TTAFrame():
    def __init__(self, net):
        # 网络搭建
        self.net = net()
        # 是否有gpu
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        # 计算参数量
        self.num_params = sum([param.nelement() for param in self.net.parameters()])

    def test_one_img_from_path(self, img, evalmode=True):
        # 测试时使用验证模式
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(img)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(img)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(img)
        else:
            return self.test_one_img_from_path_8(img)

    def test_one_img_from_path_8(self, img):
        # img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.array(img1)[:, :, ::-1]
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_4(self, img):
        """
        TTA Test Time Augmentation
        """
        # 旋转90度
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        # 长反转
        img2 = np.array(img1)[:, ::-1]
        # 宽反转
        img3 = np.array(img1)[:, :, ::-1]
        # 长宽反转
        img4 = np.array(img2)[:, :, ::-1]

        img1 = img1.transpose(0, 3, 1, 2)
        img2 = img2.transpose(0, 3, 1, 2)
        img3 = img3.transpose(0, 3, 1, 2)
        img4 = img4.transpose(0, 3, 1, 2)

        img1 = V(torch.Tensor(np.array(img1, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32) / 255.0 * 3.2 - 1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        maska = self.net.forward(img1)
        maska = maska.squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2)
        maskb = maskb.squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3)
        maskc = maskc.squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4)
        maskd = maskd.squeeze().cpu().data.numpy()
        # 四种不同角度预测结果相加（TTA）
        mask1 = maska + maskb[:, ::-1] + maskc[:, :, ::-1] + maskd[:, ::-1, ::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1, ::-1]

        return mask2

    def test_one_img_from_path_2(self, img):
        # img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = img3.transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0
        img5[:, :4, :, :] = img5[:, :4, :, :] * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0, 3, 1, 2)
        img6 = np.array(img6, np.float32) / 255.0
        img6[:, :4, :, :] = img6[:, :4, :, :] * 3.2 - 1.6
        img6 = V(torch.Tensor(img6).cuda())

        # maska = self.net.forward(img5).squeeze().cpu().data.numpy()  # .squeeze(1)
        # maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        maska = self.net.forward(img5)
        maska = maska.squeeze().cpu().data.numpy()
        maskb = self.net.forward(img6)
        maskb = maskb.squeeze().cpu().data.numpy()

        mask1 = maska + maskb[:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def test_one_img_from_path_1(self, img):
        # img = cv2.imread(path)#.transpose(2,0,1)[None]

        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None], img90[None]])
        img2 = np.array(img1)[:, ::-1]
        img3 = np.concatenate([img1, img2])
        img4 = np.array(img3)[:, :, ::-1]
        img5 = np.concatenate([img3, img4]).transpose(0, 3, 1, 2)
        img5 = np.array(img5, np.float32) / 255.0 * 3.2 - 1.6
        img5 = V(torch.Tensor(img5).cuda())

        mask = self.net.forward(img5)
        mask = mask.squeeze().cpu().data.numpy()

        mask1 = mask[:4] + mask[4:, :, ::-1]
        mask2 = mask1[:2] + mask1[2:, ::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1, ::-1]

        return mask3

    def load(self, path):
        model = torch.load(path)
        self.net.load_state_dict(model)
        self.flops, self.params = get_model_complexity_info(self.net, (3, 512, 512), as_strings=True, print_per_layer_stat=False)

def test(img, target, solver):
    _img = cv2.resize(img, (512,512))
    pred = solver.test_one_img_from_path(_img)
    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    cv2.imwrite(target, (255 * pred).astype(np.uint8))

if __name__ == '__main__':
    NAME = 'ENet'
    weight_file = os.path.join('weights', NAME, NAME+'.th')
    predict_path = os.path.join('predicts', NAME)
    if not os.path.exists(predict_path): os.makedirs(predict_path)
    # 测试流程
    solver = TTAFrame(ENet)
    # 模型参数加载
    solver.load(weight_file)

    print('Model params = {:2.1f}M'.format(solver.num_params / 1000000))
    print('FLOPS: {}, Params: {}'.format(solver.flops, solver.params))

    test_img_names = glob('DRIVE/test/images/*.tif')
    for img_name in tqdm(test_img_names):
        target = os.path.join(predict_path, os.path.basename(img_name))
        img = cv2.imread(img_name)
        test(img, target, solver)