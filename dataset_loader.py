import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image


class ImageDatasetLoader(Dataset):
    """
    加载图像数据集
    
    参数：
        type:数据集类型，可选：'MRI'、'PET'
        split:数据集划分，可选：'train'、'val'、'test'
        image_size:图像大小
        data_dir:数据集目录
        transform:自定义的数据变换
    """
    def __init__(self,types:list[str],split:str,image_size:int,data_dir:str,batch_size:int = 32,transform=None):
        self.types = types
        self.split = split
        self.image_size = image_size
        self.batch_size = batch_size
        self.transform = transform
        self.data_dir = [data_dir + '/' + type for type in types]
        self.image_paths = [] # 设置图像路径列表
        # 读取图像路径
        for d in self.data_dir:
            for (_,item) in enumerate(os.listdir(d)):
                self.image_paths.append(d + '/' + item)

    
    def default_transform(self, image):
        # 这里把原图像拉伸为image_sizeximage_size的大小，并转换为Tensor
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为Tensor
            transforms.Normalize(mean=[0.0], std=[1.0]),  # 标准化
            # transforms.Lambda(lambda x: x.view(-1)) # 将图像展平为一维向量
        ])(image)

    def __len__(self):
        return len(self.image_paths)  
        # return 5  

    def __getitem__(self, index):
        # 加载图像
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('L')  # 转为灰度图
        # # 查看图像
        # image.show()
        # 应用默认变换
        if self.transform is None:
            image = self.default_transform(image)
        else:
            image = self.transform(image)
        
        return image, image_path.split('/')[-2]  # 返回图像和标签（目录名）




if __name__ == '__main__':
    loader = ImageDatasetLoader(['AD','NC'],'train',256,'./data/MRI',32)
    loader.__getitem__(0)