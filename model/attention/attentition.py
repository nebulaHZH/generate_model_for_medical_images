# attention 机制
import torch.nn as nn
import torch
class Attention(nn.Module):
    '''
    点积注意力机制
    '''
    def __init__(self,ch=128)->None:
        super(Attention, self).__init__()
        self.linear_1 = nn.Linear(ch, ch)
        self.linear_2 = nn.Linear(ch, ch)
        self.linear_3 = nn.Linear(ch, ch)
        self.linear_final = nn.Linear(ch, ch)
    
    def forward(self,x:torch.Tensor):
        b,c,h,w = x.shape
        xt = x.view(b,c,h*w).transpose(1,2)
        key = self.linear_1(xt) 
        query = self.linear_2(xt)
        value = self.linear_3(xt)
        query = query.view(b,-1,1,c).transpose(1,2)
        key = key.view(b,-1,1,c).transpose(1,2)
        value = value.view(b,-1,1,c).transpose(1,2)

        a = nn.functional.scaled_dot_product_attention(query,key,value)
        a = a.transpose(1,2).view(b,-1,c)
        a = self.linear_final(a)
        a = a.transpose(-1,-2).reshape(b,c,h,w)

        # 加入一个residual
        return a + x