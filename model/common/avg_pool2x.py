# 完成模型的基础模块
from torch import nn
import torch
class AvgPool2x(nn.Module):
    '''
    2x2的均值池化
    '''
    def __init__(self):
        super(AvgPool2x,self).__init__()
        self.pool = nn.AvgPool2d(2,2,0)
    def forward(self,x):
        return self.pool(x)
    
# 测试
if __name__ == '__main__':
    a = torch.randn(1, 1, 256, 256)
    print("原始尺寸：",a.shape,"原始数据：",a)
    avgpool = AvgPool2x().forward(a)
