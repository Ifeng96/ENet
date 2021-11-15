#!/usr/bin/python3
# -*- coding: utf-8 -*-
#########################
#    python第三方库
#########################
# pytorch库：构建深度学习网络及训练测试等
import torch
# 数据加载库：用于送入数据到网络中
from torch.utils.data import DataLoader
# 图像库：保存训练过程中的图像
import cv2
# 操作系统库：操作文件与目录
import os
# 数据处理库：处理数据
import numpy as np
# 时间库：用于记录时间
from time import time
# 进度条库：实时显示长训练的进度
from tqdm import tqdm

#########################
#   导入自己实现的模块
#########################
# 网络结构
from networks import ENet
# 训练流程架构
from utils.framework import MyFrame
# 数据读取载入
from utils.data import ImageFolder
# 评价指标
from utils.metric import metric

#########################
#   路径设置
#########################
# 训练保存名称，通过NAME进行区分
NAME = 'ENet'
# 模型权重保存的路径，文件夹下不存在则新建
weight_path = os.path.join('weights/', NAME)
if not os.path.exists(weight_path): os.makedirs(weight_path)
# 模型训练过程预测结果保存的路径
pred_path = os.path.join('pred_log', NAME)
if not os.path.exists(pred_path): os.makedirs(pred_path)
# 模型训练过程中的日志记录路径
if not os.path.exists('logs'): os.makedirs('logs')
# 日志文件打开记录
mylog = open('logs/'+NAME+'.log','a')

#########################
#   超参数
#########################
BATCHSIZE_PER_CARD = 4  # batch_size,单卡gpu每次送入几张图片
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD # 真实batchsize为4 * gpu卡的数量 （一般单卡则为4*1=4）
learning_rate = 5e-4    # 权重更新的步长
class_num = 1 # 类别数（两类设置为1，多余两类则设置为真实类别数）

#########################
#   训练流程
#########################
# 训练流程搭建，具体见utils.framework.MyFrame类
solver = MyFrame(ENet, class_num, metric, learning_rate)
# 模型参数量打印及记录
print('Model params = {:2.1f}M'.format(solver.num_params / 1000000), file=mylog)
print('Model params = {:2.1f}M'.format(solver.num_params / 1000000))

#########################
#   数据集构建
#########################
# 数据的输入尺寸
SHAPE = (512, 512)
# 训练数据集路径
train_path = 'DRIVE/training'
dataset = ImageFolder(train_path, 'training', shape=SHAPE)
data_loader = DataLoader(dataset, batchsize, True, drop_last=True, pin_memory=True)
# 验证数据集路径
val_path = 'DRIVE/test'
dataset_val = ImageFolder(val_path, 'test', shape=SHAPE)
data_loader_val = DataLoader(dataset_val, batchsize, False, drop_last=True, pin_memory=True)
# 训练集、验证集数量
num_train, num_val = len(data_loader), len(data_loader_val)

#########################
#   模型训练验证
#########################
# 时间开始，记录当前时刻点
tic = time()
# 训练过程中记录精度未继续下降的轮数，初始化为0
no_optim = 0
# 总训练轮数，即将全部数据训练一边的次数，这个可以根据实际数据改变
# 不过可以设置稍微大一点，模型设置有early stop，不一定训练这么多轮才结束
total_epoch = 200
# 训练过程中的最优损失，初始化为100
train_epoch_best_loss = 100.
# 验证过程的iou，初始化为0
val_epoch_iou = 0

# 优化器梯度设置
solver.optimizer.zero_grad()
solver.optimizer.step()
# 开始训练
for epoch in range(1, total_epoch + 1):
    # 学习率衰减，根据验证iou进行衰减
    solver.scheduler_warmup.step_ReduceLROnPlateau(val_epoch_iou, epoch)
    #########################
    #       训练模式
    #########################
    solver.net.train()
    # 训练损失及iou，设置为0
    train_epoch_loss, train_epoch_iou = 0, 0
    # 数据集遍历，tqdm可以实时显示数据集遍历进度
    bar = tqdm(data_loader)
    # 循环读取数据
    for data in bar:
        # 将读取的数据送入cuda中
        solver.set_input(*data)
        # 训练优化，返回损失
        train_loss = solver.optimize()
        # 记录当前epoch中损失总和
        train_epoch_loss += train_loss
        # 记录当前epoch中miou总和
        train_epoch_iou += solver.metric.miou
        # 将损失与miou显示在进度条上
        bar.set_description('loss:{:.3f}, miou:{:.3f}'.format(train_loss, solver.metric.miou))
    # 计算epoch的平均损失与iou
    train_epoch_loss /= num_train
    train_epoch_iou /= num_train

    #########################
    #训练中原图、标签、预测图的保存
    #########################
    # 当前预测结果图像的数量（一般等于batchsize）
    N = solver.pred.shape[0]
    # NCHW --> NHWC，将通道数翻转到最后一维
    train_pred = np.transpose(solver.pred, (0, 2, 3, 1))
    train_mask = np.transpose(solver.mask, (0, 2, 3, 1))
    # 经过预处理的图像恢复到原图0-1
    img = (np.transpose(data[0].cpu().data.numpy(), (0, 2, 3, 1)) + 1.6) / 3.2
    # 将当前预测的几张图像保存在pred_path中
    for i in range(N):
        # 原图、预测图、标签拼接起来，预测图与标签扩展为3通道
        train_img = np.concatenate((train_pred[i], train_mask[i]), axis=1).repeat(3, axis=-1)
        train_img = np.concatenate((img[i], train_img), axis=1)
        cv2.imwrite(pred_path + '/train_epoch{}_{}.bmp'.format(epoch, i), 255 * train_img)

    #########################
    #       验证模式
    #########################
    solver.net.eval()
    # 验证iou，初始设置为0
    val_epoch_iou = 0
    # 遍历验证数据集
    for data in data_loader_val:
        # 将读取的数据送入cuda中
        solver.set_input(*data)
        # 进行预测评估
        solver.evaluate()
        # 记录当前epoch中miou总和
        val_epoch_iou += solver.metric.miou
    # 计算epoch的平均iou
    val_epoch_iou /= num_val

    #########################
    # 验证中原图、标签、预测图的保存
    #########################
    # 当前预测结果图像的数量（一般等于batchsize）
    N = solver.pred.shape[0]
    # NCHW --> NHWC，将通道数翻转到最后一维
    val_pred = np.transpose(solver.pred, (0, 2, 3, 1))
    val_mask = np.transpose(solver.mask, (0, 2, 3, 1))
    # 经过预处理的图像恢复到原图0-1
    img = (np.transpose(data[0].cpu().data.numpy(), (0, 2, 3, 1)) + 1.6) / 3.2
    # 将当前预测的几张图像保存在pred_path中
    for i in range(N):
        # 原图、预测图、标签拼接起来，预测图与标签扩展为3通道
        val_img = np.concatenate((val_pred[i], val_mask[i]), axis=1).repeat(3, axis=-1)
        val_img = np.concatenate((img[i], val_img), axis=1)[:,:,::-1]
        cv2.imwrite(pred_path + '/val_epoch{}_{}.bmp'.format(epoch, i), 255 * val_img)

    #########################
    #    日志保存及打印
    #########################
    # 日志保存
    print('********', file=mylog)
    cost_time = int(time() - tic)
    print('epoch:', epoch, '    time:', cost_time//3600, ' h, ', cost_time%3600//60, ' min', file=mylog)
    print('learning rate: ', solver.scheduler_warmup.get_lr(), file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)
    print('train_iou:', train_epoch_iou, file=mylog)
    print('val_iou:', val_epoch_iou, file=mylog)
    print('SHAPE:', SHAPE, file=mylog)
    # 日志打印
    print('*******')
    print('epoch:', epoch, '    time:', cost_time//3600, ' h, ', cost_time%3600//60, ' min')
    print('learning rate: ', solver.scheduler_warmup.get_lr())
    print('train_loss:',train_epoch_loss)
    print('train_iou:', train_epoch_iou)
    print('val_iou:', val_epoch_iou)
    print('SHAPE:',SHAPE)

    #########################
    #    模型早停设置
    #########################
    if train_epoch_loss >= train_epoch_best_loss:
        # 损失未继续下降则未优化轮数+1
        no_optim += 1
    else:
        # 否则清0
        no_optim = 0
        # 将当前损失保存为最优损失
        train_epoch_best_loss = train_epoch_loss
        # 保存模型权重
        solver.save(weight_path+'/'+NAME+'.th')
        # 记录最后一次模型保存的epoch数
        last_saved = epoch
    # 如果连续10轮未进行优化则早停，退出训练
    if no_optim > 10:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    # 写入日志文件
    mylog.flush()

# 训练完成
print('Finish!', file=mylog)
print('Finish!')
mylog.close()
