import torch
from dataset_loader import ImageDatasetLoader
from model.VAE import VAEDecoder, VAEEncoder
from utils.display import Display
from torch.utils.data import DataLoader

def VAETest(num_epochs:int):
    # 1. 初始化模型参数
    vae_encoder = VAEEncoder(input_dim=256*256, hidden_dim=1024, latent_dim=64)
    vae_decoder = VAEDecoder(latent_dim=64, hidden_dim=1024, output_dim=64*32*32,batch_size=1)
    # 2. 加载模型参数
    checkpoint = torch.load(f'./pts/vae_epoch_{num_epochs-1}.pth')
    # 3. 加载模型参数到模型中
    vae_decoder.load_state_dict(checkpoint['decoder_state_dict'])
    data_loader = ImageDatasetLoader(
        types=['glioma', 'meningioma'], 
        split='train',
        image_size=256,
        batch_size=64,
        data_dir='./data/tumor/Training',
        transform=None
    )
    data_loader = DataLoader(dataset=data_loader, batch_size=1, shuffle=True)
    z = data_loader.__iter__().__next__()
    # # 随机生成一个潜空间的样本
    # z = torch.randn(1, 1,256,256)
    z = z[0]
    Display(tensor=z,title="Original Image")
    z = vae_encoder(z)[0]
    # 生成样本
    generate_image = vae_decoder(z)

    # 换成正确的尺寸
    generate_image = generate_image.view(1, 256, 256)

    # 展示图片
    Display(tensor=generate_image,title="Generated Image")


if __name__ == '__main__':
    VAETest(50)