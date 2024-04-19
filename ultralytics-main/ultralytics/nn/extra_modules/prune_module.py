import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv, DWConv, RepConv, GhostConv, autopad
from ..modules.block import *

# 没有购买yolov8项目需要注释以下
from .block import Faster_Block, MBConv, RepNCSP

class C2f_infer(nn.Module):
    # CSP Bottleneck with 2 convolutions For Infer
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c1, self.c2 = 0, 0
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c1, self.c2), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2f_EMBC_infer(nn.Module):
    # CSP Bottleneck with 2 convolutions For Infer
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c1, self.c2 = 0, 0
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MBConv(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c1, self.c2), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2f_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    '''`C2f_v2`这个Python类继承自PyTorch的`nn.Module`，它定义了一个神经网络模块，这个模块实现了一个包含两个卷积层的CSP瓶颈（Cross Stage Partial Networks，CSPNet）结构。这个结构通常用于卷积神经网络以提升性能和效率。

这里是类`C2f_v2`的细节解释：

- `__init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5)`：这是构造函数，用来初始化这个神经网络模块。
    - `c1`：输入通道数。
    - `c2`：输出通道数。
    - `n`：CSP瓶颈重复的次数。
    - `shortcut`：是否在瓶颈结构中使用残差连接。
    - `g`：卷积的组数，用于分组卷积。
    - `e`：扩展因子，用于调整隐藏层通道数。

- 类变量：
    - `self.c`：隐藏层通道数，通过输出通道数`c2`和扩展因子`e`计算得到。
    - `self.cv0`和`self.cv1`：第一层卷积操作，将输入的`c1`通道转换为隐藏层通道`self.c`。
    - `self.cv2`：第二层卷积操作，融合所有前面的层的输出，将通道数从`(2 + n) * self.c`变为输出通道数`c2`。
    - `self.m`：一个瓶颈层的模块列表，每个瓶颈层将输入通道`self.c`处理后输出同样通道数`self.c`。

- `forward(self, x)`：这是前向传播函数，定义了数据通过网络层的流向。
    - 首先，通过`self.cv0`和`self.cv1`对输入的`x`执行两个分支的卷积操作，得到两个输出`y`。
    - 接着，通过循环将输入`x`传递给瓶颈层`self.m`中的每一个模块，每次循环后的输出都会添加到`y`列表中。
    - 最后，将`y`列表中的所有输出进行连接（concatenate），然后通过`self.cv2`执行最终的卷积操作，输出最终的特征图。

这个类可以成为一个复杂神经网络架构中的一部分，通常用于图像识别、目标检测等任务中。 （GPT-4）'''

class C2f_Faster_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2f_EMBC_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(MBConv(self.c, self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class RepNCSPELAN4_v2(nn.Module):
    # csp-elan
    def __init__(self, c1, c2, c3, c4, c5=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = c3//2
        self.cv0 = Conv(c1, c3 // 2, 1, 1)
        self.cv1 = Conv(c1, c3 // 2, 1, 1)
        self.cv2 = nn.Sequential(RepNCSP(c3 // 2, c4, c5), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepNCSP(c4, c4, c5), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))