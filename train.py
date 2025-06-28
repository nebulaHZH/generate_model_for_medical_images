from datetime import time
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from dataset_loader import ImageDatasetLoader
from model.VAE import VAE, VAEEncoder,VAEDecoder

###########################################
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
###########################################


data_loader = ImageDatasetLoader(
    types=['glioma', 'meningioma'], 
    split='train',
    image_size=256,
    batch_size=64,
    data_dir='./data/tumor/Training',
    transform=None
)

data_loader = DataLoader(dataset=data_loader, batch_size=4, shuffle=True)
# 修改潜在空间维度为一个更合理的值，例如64
vae_encoder = VAEEncoder(input_dim=256, hidden_dim=1024, latent_dim=64).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
vae_decoder = VAEDecoder(latent_dim=64, hidden_dim=1024, output_dim=64*32*32).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# 更新优化器以同时优化编码器和解码器的参数
optimizer = optim.Adam(list(vae_encoder.parameters()) + list(vae_decoder.parameters()), lr=1e-4)

def train():
    loss = torch.tensor(0.0)  # 修改这里
    for epoch in range(num_epochs):
        # 使用 tqdm 包裹 data_loader
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")
        for i, (image, _) in progress_bar:
            # # -1. 先把图片归一化到0-1之间
            # image = image.float() / 255.0
            image = image.to(device)  # 将图像移动到 GPU
            # 0. 清空梯度
            optimizer.zero_grad()
            # 1. 使用VAE编码器进行编码到潜空间
            z,mu, logvar = vae_encoder(image)
            x = vae_decoder(z)
            # 3. 计算损失
            loss = VAE.loss_function(x, image,mu, logvar)
            # 4. 反向传播
            loss.backward()
            # 5. 优化器更新参数
            optimizer.step()
    # 只保留最新的模型
    torch.save({
        'encoder_state_dict': vae_encoder.state_dict(),
        'decoder_state_dict': vae_decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    },f'./pts/vae_epoch_{epoch}.pth')

if __name__ == "__main__":
    train()

