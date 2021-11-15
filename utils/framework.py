#!/usr/bin/python3
# -*- coding: utf-8 -*-
# pytorch相关库
import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.optim.sgd import SGD

# 学习率衰减实现及损失计算实现
from utils.lr_scheduler import GradualWarmupScheduler
from loss import dice_bce_loss

# 训练框架类
class MyFrame():
    def __init__(self, net, class_num, metric, lr=2e-4, evalmode = False):
        """
        :param net: 网络框架
        :param class_num: 类别数
        :param metric: 评价指标
        :param lr: 学习率
        :param evalmode: 是否为验证模式
        """
        # 网络搭建
        self.net = net(class_num)
        # 如果存在gpu则导入到gpu中，存在多个gpu则分发到每个gpu上
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        # 网络的参数量计算
        self.num_params = sum([param.nelement() for param in self.net.parameters()])
        # 优化器选择，一般建议adam，基本都能适用
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr, weight_decay=1e-5)
        # 学习率衰减，每次衰减0.5倍，三轮没有提升则进行衰减，最小学习率5e-7，max代表参考指标是增加（见train.py中参考指标取的是val_iou）
        # 即val_iou连续3轮未增加则学习率衰减未0.5倍，最低未5e-7
        self.relr = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=3, min_lr=5e-7, mode='max')
        # warmup，刚开始的5轮进行warmup
        self.scheduler_warmup = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=5, after_scheduler=self.relr)
        # 损失函数实例化
        self.dice_bce = dice_bce_loss()
        # 评价指标实例化
        self.metric = metric(False)
        # 当前学习率
        self.old_lr = lr
        # 验证模式下bn设置不同
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, img_id=None, volatile=False):
        """
        将数据导入cuda
        :param img_batch: 图像batch [batch_size,channel,height,width]
        :param mask_batch: 标签batch [batch_size,class_num,height,width]
        :param img_id: 图像名称 [batch_size]
        :param volatile:
        """
        if torch.cuda.is_available():
            img_batch = img_batch.cuda()
            mask_batch = mask_batch.cuda()
        self.img = V(img_batch, volatile=volatile)
        self.mask = None
        if mask_batch is not None:
            self.mask = V(mask_batch, volatile=volatile)
        self.img_id = img_id

    def optimize(self):
        """
        模型参数反向传播优化
        """
        # 梯度设为0，后面计算预测结果
        self.optimizer.zero_grad()
        # 前向输出预测结果
        self.pred = self.net.forward(self.img)
        # 标签预测计算苏纳是
        loss = self.dice_bce(self.mask, self.pred)
        # 预测及标签拷贝到cpu上
        self.pred, self.mask = self.pred.cpu().data.numpy(), self.mask.cpu().data.numpy()
        # 计算评价指标
        self.metric(self.pred.copy(), self.mask)
        # 反向计算梯度
        loss.backward()
        # 参数优化
        self.optimizer.step()
        # 返回损失
        return loss.item()

    def evaluate(self):
        # 梯度设为0
        self.optimizer.zero_grad()
        # 不计算梯度
        with torch.no_grad():
            self.pred = self.net.forward(self.img)
        # 预测及标签拷贝到cpu上
        self.pred, self.mask = self.pred.cpu().data.numpy(), self.mask.cpu().data.numpy()
        # 计算评价指标
        self.metric(self.pred.copy(), self.mask)

    def save(self, path):
        """
        保存模型参数到path下
        """
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        """
        加载模型参数
        """
        self.net.load_state_dict(torch.load(path))
    
    # def update_lr(self, new_lr, mylog, factor=False):
    #     if factor:
    #         new_lr = self.old_lr / new_lr
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] = new_lr
    #
    #     print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
    #     print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
    #     self.old_lr = new_lr
