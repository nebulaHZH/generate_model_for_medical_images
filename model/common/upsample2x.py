from torch import nn
import torch

class Upsample2x(nn.Module):
    '''
    2倍上采样
    '''
    def __init__(self,
                 ch_input:int,
                 ch_output:int,
                 conv_t:bool = True)->None:
        super(Upsample2x, self).__init__()
        # 选择使用conv transpose还是卷积
        self.conv_t = conv_t
        if conv_t:
            self.up = nn.ConvTranspose2d(ch_input, ch_output, kernel_size=4, stride=2, padding=1,bias=False)
        else:
            self.conv = nn.Conv2d(ch_input, ch_output, kernel_size=1, stride=1, padding=0) # 1x1 卷积保持插值与反卷积的通道数一致
        self.norm = nn.BatchNorm2d(ch_output)
        self.silu = nn.SiLU()
    def forward(self, x)-> torch.Tensor:
        if self.conv_t:
            upsample = self.up(x)
        else:
            upsample = self.conv(nn.functional.interpolate(x, scale_factor=2, mode='nearest'))
        return self.silu(self.norm(upsample))




if __name__ == '__main__':
    a = torch.randn(1, 1, 256, 256)
    print("原始尺寸：",a.shape,"原始数据：",a)
    upsample = Upsample2x(ch_input=1,ch_output=32).forward(a)
    print("上采样后尺寸：",upsample.shape)
