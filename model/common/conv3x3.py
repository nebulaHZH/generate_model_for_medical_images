from torch import nn
import torch
class Conv3x3(nn.Module):
    '''
    3x3卷积层
    
    ch_input 输入通道数
    ch_output 输出通道数
    '''
    def __init__(self,
                 ch_input:int,
                 ch_output:int,
                 act:bool=False)->None:
        super(Conv3x3,self).__init__()
        self.act = act
        self.conv = nn.Conv2d(in_channels=ch_input,out_channels=ch_output,kernel_size=3,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(ch_output)
        if act:
            self.silu = nn.SiLU()
    def forward(self,x:torch.Tensor) -> torch.Tensor:
        output = self.bn(self.conv(x))
        if self.act:
            output = self.silu(output)
        return output



if __name__ == '__main__':
    a = torch.randn(1, 1, 256, 256)
    print("原始尺寸：",a.shape,"原始数据：",a)
    conv3x3 = Conv3x3(ch_input=1,ch_output=32).forward(a)
    print("3x3卷积后尺寸：",conv3x3.shape)
