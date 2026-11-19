# nn.Conv1d
# nn.Conv2d
# nn.Conv3d

# 卷积神经网络不会改变输入的维度

#########################################################
# Conv1d（一维卷积）
# 输入形状：
# (N, C, L)
# 卷积核沿着一个方向移动，适合时间序列、传感器数据、音频波形等。
# Conv2d（二维卷积）
# 输入形状：
# (N, C, H, W)
# 卷积核沿着高度和宽度两个方向移动，适合图片。
# Conv3d（三维卷积）
# 输入形状：
# (N, C, D, H, W)
# 卷积核沿着三个空间维度移动，适合视频、CT 医学影像、三维体数据等。



#|-----------------二维卷积--------------------------|
#########################################################
import torch
import torch.nn as nn

x = torch.randn(2,3,32,32)

conv2 = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1
)# 2d处理的是二维的数据（batch_size,channels,height,width）
# out_channels输出通道数：可以表示卷积层需要提取多少组特征
y = conv2(x)

print(y.shape)

#|----------------------一维卷积----------------------|
#########################################################
import torch
import torch.nn as nn

x = torch.randn(2, 3, 10)
conv1 = nn.Conv1d(
    in_channels=3,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1
)
y = conv1(x)

print(y.shape)

#|----------------------三维卷积----------------------|
#########################################################
import torch
import torch.nn as nn

x = torch.randn(2, 3, 10,10,10)
conv3 = nn.Conv3d(
    in_channels=3,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1
)
y = conv3(x)

print(y.shape)