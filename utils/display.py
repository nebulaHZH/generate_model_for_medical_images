

from matplotlib import pyplot as plt
import torch

class Display():
    """图片显示类

    """
    
    def __init__(self,tensor:torch.Tensor,title:str = 'display image'):
        super().__init__()
        self.tensor = tensor
        self.title = title
        self.show_image()

    def show_image(self):
        tensor = self.tensor.detach().cpu().squeeze()
        # 把灰度图转换为0-255之间的整数，取消归一化
        tensor = (tensor * 255).clamp(0, 255).byte().numpy()
        plt.figure(figsize=(10, 10))
        plt.imshow(tensor, cmap='gray')
        plt.title(self.title)
        plt.axis('off')
        plt.show()