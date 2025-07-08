# 数据集加载
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image,ImageReadMode
from torchvision.transforms import v2,InterpolationMode
from torch.utils.data import DataLoader

class ImageData(Dataset):
    def __init__(self,
                image_dir:str,
                image_size:int,
                gray_scale:bool = False,
                device:torch.device = torch.device('cpu'),
                load_all:bool = False,)->None:
        self.image_paths = [f.path for f in os.scandir(image_dir) if os.path.isfile(f) and os.path.splitext(f)[-1].lower() in [".jpg", ".png", ".jpeg", ".bmp"]]
        self.target_size = image_size
        self.transforms = v2.Compose([
            v2.Resize((self.target_size, self.target_size),InterpolationMode.BILINEAR),
            v2.PILToTensor(),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize([1275],[127.5])
        ])
        self.device = device
        self.load_all = load_all
        if load_all:
            self.images = [self.transforms(read_image(path,mode=ImageReadMode.GRAY if not gray_scale else ImageReadMode.GRAY).float()) for path in self.image_paths]
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index:int)->torch.Tensor:
        if self.load_all:
            return self.images[index]
        else:
            image = read_image(self.image_paths[index],mode=ImageReadMode.GRAY).float()
            image:torch.Tensor = self.transforms(image)
            return image.to(self.device)

# 测试使用
if __name__ == '__main__':
    dataset = ImageData(
        image_dir='D:\\0-nebula\\dataset\\Havard\\MyDatasets\\CT-MRI\\test\\CT',
        image_size=256,
        gray_scale=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        load_all=False
    )
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    # 显示一个batch的形状
    image:torch.Tensor = next(iter(data_loader)) 
    image = image[0]
    # 展示图片
    import matplotlib.pyplot as plt
    plt.imshow(image.cpu().permute(1,2,0).numpy(), cmap='gray')
    plt.show()
