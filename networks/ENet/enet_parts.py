import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    def __init__(self,in_channels,out_channels,bias=False,relu=True):
        super().__init__()
        if relu:
            activation=nn.ReLU
        else:
            activation=nn.PReLU
        # 3x3卷积分支,(输出通道-3)是为了方便与maxpooling通道的卷积结果拼接后,无需再进行变换
        self.main_branch=nn.Conv2d(in_channels,out_channels-3,kernel_size=3,
                                                        stride=2,padding=1,bias=bias)
        self.ext_branch=nn.MaxPool2d(3,stride=2,padding=1)
        self.batch_norm=nn.BatchNorm2d(out_channels)
        self.out_activation=activation()

    def forward(self,x):
        main=self.main_branch(x)
        ext=self.ext_branch(x)
        # 通道维度上叠加
        out=torch.cat((main,ext),1)
        out=self.batch_norm(out)
        return self.out_activation(out)

# 无MaxPooling与Padding,不改变宽高
class RegularBottleneck(nn.Module):
    def __init__(self,channels,internal_ratio=4,kernel_size=3,padding=0,dilation=1,
                             asymmetric=False,dropout_prob=0,bias=False,relu=True):
        super().__init__()
        internal_channels=channels//internal_ratio
        if relu:
            activation=nn.ReLU
        else:
            activation=nn.PReLU

        # 1x1 convolution, 减小通道数
        self.conv1=nn.Sequential(
            nn.Conv2d(channels,internal_channels,kernel_size=1,stride=1,bias=bias),
            nn.BatchNorm2d(internal_channels),activation())
        # asymmetric为true则为非对称卷积,使用kx1卷积和1xk卷积代替kxk卷积
        # asymmetric为False,dilation=1则为普通卷积(kxk),dilation>1则为空洞卷积
        if asymmetric:
            self.conv2=nn.Sequential(
             nn.Conv2d(internal_channels,internal_channels,kernel_size=
          (kernel_size,1),stride=1,padding=(padding,0),dilation=dilation,bias=bias),
             nn.BatchNorm2d(internal_channels),activation(),
             nn.Conv2d(internal_channels,internal_channels,kernel_size=
          (1,kernel_size),stride=1,padding=(0,padding),dilation=dilation,bias=bias),
             nn.BatchNorm2d(internal_channels),activation())
        else:
            self.conv2=nn.Sequential(
                  nn.Conv2d(internal_channels,internal_channels,kernel_size=
                  kernel_size,stride=1,padding=padding,dilation=dilation,bias=bias),
                   nn.BatchNorm2d(internal_channels),activation())
        # 1x1 convolution, 扩大通道数
        self.conv3=nn.Sequential(
             nn.Conv2d(internal_channels,channels,kernel_size=1,stride=1,bias=bias),
             nn.BatchNorm2d(channels),activation())
        self.regul=nn.Dropout2d(p=dropout_prob)
        # 叠加后进行一次激活
        self.out_activation=activation()

    def forward(self,x):
        # Main branch shortcut
        main=x
        # Extension branch
        ext=self.conv1(x)
        ext=self.conv2(ext)
        ext=self.conv3(ext)
        ext=self.regul(ext)
        # Add main and extension branches
        out=main+ext
        return self.out_activation(out)

# 主通道上有下采样
class DownsamplingBottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,internal_ratio=4,return_indices=False
                                               ,dropout_prob=0,bias=False,relu=True):
        super().__init__()
        # Store parameters that are needed later
        self.return_indices=return_indices
        internal_channels=in_channels//internal_ratio
        if relu:
            activation=nn.ReLU
        else:
            activation=nn.PReLU

        # 主通道上maxpooling,kernel_size=2,stride=2使得特征图大小减半
        # return_indices=True，会返回输出最大值的序号，对于上采样操作会有帮助
        self.main_max1=nn.MaxPool2d(2,stride=2,return_indices=return_indices)

        # 2x2 convolution,使用步长为2的卷积,使特征图大小减半
        self.ext_conv1=nn.Sequential(
            nn.Conv2d(in_channels,internal_channels,kernel_size=2,stride=2,
                                                                   bias=bias),
            nn.BatchNorm2d(internal_channels),activation())
        # 3x3 Convolution, 普通卷积, 不改变通道数
        self.ext_conv2=nn.Sequential(
             nn.Conv2d(internal_channels,internal_channels,kernel_size=3,
                                                stride=1,padding=1,bias=bias),
             nn.BatchNorm2d(internal_channels),activation())
        # 1x1 convolution, 扩大通道数
        self.ext_conv3=nn.Sequential(
            nn.Conv2d(internal_channels,out_channels,kernel_size=1,stride=1,
                                                                  bias=bias),
            nn.BatchNorm2d(out_channels),activation())
        # 对特征图使用dropout
        self.ext_regul=nn.Dropout2d(p=dropout_prob)

        self.out_activation=activation()

    def forward(self,x):
        # return_indices=True,main记录maxpool结果,max_indices最大值索引
        if self.return_indices:
            main,max_indices=self.main_max1(x)
        else:
            main=self.main_max1(x)

        # Extension branch
        ext=self.ext_conv1(x)
        ext=self.ext_conv2(ext)
        ext=self.ext_conv3(ext)
        ext=self.ext_regul(ext)

        # 对main通道进行padding,main的batch_size、h、w与ext的batch_size、h、w相同
        # 但是,可能因为求商(//)导致main与ext的channel不一致,对其补零,使其方便相加
        n,ch_ext,h,w=ext.size()
        ch_main=main.size()[1]
        padding=torch.zeros(n,ch_ext-ch_main,h,w)
        # 判断main是否在cuda上,如果在,将padding移至cuda,不然无法concat
        if main.is_cuda:
            padding=padding.cuda()
        main=torch.cat((main,padding),1)

        # Add main and extension branches
        out=main+ext
        return self.out_activation(out),max_indices

# 主通道上有上采样
class UpsamplingBottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,internal_ratio=4,dropout_prob=0,
                 bias=False,relu=True):
        super().__init__()
        internal_channels=in_channels//internal_ratio
        if relu:
            activation=nn.ReLU
        else:
            activation=nn.PReLU

        # 1x1 convolution,变换通道数
        self.main_conv1=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=bias),
            nn.BatchNorm2d(out_channels))
        # MaxUnpool2d扩大尺寸
        self.main_unpool1=nn.MaxUnpool2d(kernel_size=2)

        # 1x1 convolution
        self.ext_conv1=nn.Sequential(
            nn.Conv2d(in_channels,internal_channels,kernel_size=1,bias=bias),
            nn.BatchNorm2d(internal_channels),activation())
        # 反卷积参数kernel_size=2,stride=2,尺寸扩大一倍
        self.ext_tconv1=nn.ConvTranspose2d(internal_channels,internal_channels,
                                              kernel_size=2,stride=2,bias=bias)
        self.ext_tconv1_bnorm=nn.BatchNorm2d(internal_channels)
        self.ext_tconv1_activation=activation()
        # 1x1 convolution
        self.ext_conv2=nn.Sequential(
             nn.Conv2d(internal_channels,out_channels,kernel_size=1,bias=bias),
             nn.BatchNorm2d(out_channels),activation())
        # 对特征图使用dropout
        self.ext_regul=nn.Dropout2d(p=dropout_prob)

        self.out_activation=activation()

    def forward(self,x,max_indices,output_size):
        # Main branch shortcut
        main=self.main_conv1(x)
        main=self.main_unpool1(main,max_indices,output_size=output_size)

        # Extension branch
        ext=self.ext_conv1(x)
        ext=self.ext_tconv1(ext,output_size=output_size)
        ext=self.ext_tconv1_bnorm(ext)
        ext=self.ext_tconv1_activation(ext)
        ext=self.ext_conv2(ext)
        ext=self.ext_regul(ext)

        # Add main and extension branches
        out=main+ext
        return self.out_activation(out)
