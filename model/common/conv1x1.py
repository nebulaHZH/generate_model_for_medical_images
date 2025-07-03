from torch import nn
import torch
class Conv1x1(nn.Module):
    '''
    1x1卷积

    ch_input 输入通道数
    ch_output 输出通道数
    '''
    def __init__(self, ch_input:int, ch_output:int):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(ch_input, ch_output, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(ch_output)
        self.silu = nn.SiLU()
    def forward(self, x)-> torch.Tensor:
        return self.silu(self.bn(self.conv(x)))



if __name__ == '__main__':
    a = torch.randn(1, 1, 256, 256)
    print("原始尺寸：",a.shape,"原始数据：",a)
    conv1x1 = Conv1x1(ch_input=1,ch_output=32).forward(a)
    print("1x1卷积后尺寸：",conv1x1.shape)