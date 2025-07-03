# 基础模型类
import torch
from thop import profile



class Model(torch.nn.Module):
    def __init__(self, ch_input:int, image_size:int,device:torch.device) -> None:
        super(Model, self).__init__()
        self.ch_input = ch_input
        self.image_size = image_size
        self.device = device
    def model_thops(self):
        with torch.no_grad():
            x = torch.randn(1, self.ch_input, self.image_size, self.image_size)
            t = torch.tensor([1000])
            macs, params = profile(self, inputs=(x, t))  # type: ignore
            print("模型信息:", "MACs", macs, "Params", params)

    def to_onnx(self, path: str):
        with torch.no_grad():
            x = torch.randn(1, self.ch_input, self.image_size, self.image_size).to(self.device) #  创建一个随机张量，形状为(1, self.ch_input, self.image_size, self.image_size)，并将其移动到self.device指定的设备上
            t = torch.tensor([1000]).to(self.device)
            torch.onnx.export(self, (x, t), path)