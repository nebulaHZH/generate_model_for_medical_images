import torch
from torch.utils.data import DataLoader
from load import ImageData



dataset = ImageData(
    image_dir='D:\\0-nebula\\dataset\\Havard\\MyDatasets\\CT-MRI\\test\\CT',
    image_size=256,
    gray_scale=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    load_all=False
)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)