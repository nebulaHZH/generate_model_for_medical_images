# 空间注意力模块
import torch.nn as nn
import torch
# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 全局平均池化（通道维度）
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 全局最大池化（通道维度）
        spatial_features = torch.cat([avg_out, max_out], dim=1)  # 拼接为H×W×2
        spatial_weights = self.sigmoid(self.conv(spatial_features))  # 卷积生成H×W×1的权重
        return x * spatial_weights  # 空间加权