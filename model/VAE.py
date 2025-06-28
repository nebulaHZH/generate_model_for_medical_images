import torch
import torch.nn as nn

# 定义编码器
class VAEEncoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 latent_dim,
                 input_channels=1,
        ):
        """
        VAE编码器（使用CNN）
        
        参数：
            input_dim: 输入维度 | hidden_dim: 隐藏层维度 | latent_dim: 潜在空间维度
        返回：
            z: 潜在空间的向量 | mu: 潜在空间的均值 | logvar: 潜在空间的对数方差
        """
        super(VAEEncoder, self).__init__()

        """"这里的输入 和 输出 的 图像尺寸减半的，因为我们使用了padding=2，卷积核大小为3x3，步长为1。"""
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels= 16, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        ) # 256*256 -> 128*128
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels= 16, out_channels= 32, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        ) # 128*128 -> 64*64
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        ) # 64*64 -> 32*32
        """ 卷积核特征提取后是一个四维的张量，形状为(batch_size, channels, height, width)  需要将其展平为一维向量，输入到全连接层 """
        self.fc_layer_1 = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(64*32*32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc_layer_2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim) 
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        """ x的形状为(batch_size, channels, height, width) """
        # x : (4,1,256,256)
        conv_out_1 = self.conv_layer_1(x)   # (4,16,128,128)
        conv_out_2 = self.conv_layer_2(conv_out_1) # (4,64,64,64)
        conv_out_3 = self.conv_layer_3(conv_out_2) # (4,256,32,32)
        h = self.fc_layer_1(conv_out_3) # (4,hidden_dim)
        h = self.fc_layer_2(h)  # (4,hidden_dim)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar) #编码器输出的潜在变量的对数方差。
        return z,mu, logvar
    

class VAEDecoder(nn.Module):
    """
    解码器
    
    参数：
        latent_dim: 潜在空间维度 | hidden_dim: 隐藏层维度 | output_dim: 输出维度
    返回：
        x_recon: 重构的输入
    """
    def __init__(self, latent_dim, hidden_dim, output_dim = 256,batch_size:int = 4):
        super(VAEDecoder, self).__init__()
        self.batch_size = batch_size

        self.deconv_input_shape = (64, 32, 32)  # 定义反卷积层输入的形状
        
        self.fc_layer_1 = nn.Sequential(
            nn.Linear(latent_dim, 64 * 32 * 32),  # 直接映射到卷积层所需的特征图尺寸
            nn.ReLU(),
        )

        self.deconv_layer_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),  # 加速训练
            nn.ReLU(),
        ) # 32*32 -> 64*64
        self.deconv_layer_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),  # 加速训练
            nn.ReLU(),
        ) # 64*64 -> 128*128
        self.deconv_layer_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),  # 输出层，使用Sigmoid激活函数将输出限制在[0, 1]之间
        ) # 128*128 -> 256*256

    def forward(self, z):
        h = self.fc_layer_1(z)
        h = h.view(z.size(0), *self.deconv_input_shape)  # 显式重塑为四维张量

        x = self.deconv_layer_1(h)  # 上采样
        x = self.deconv_layer_2(x)  
        x_recon = self.deconv_layer_3(x)
        return x_recon
    

class VAE(nn.Module):
    """
    变分自编码器
    
    参数：
        input_dim: 输入维度 | hidden_dim: 隐藏层维度 | latent_dim: 潜在空间维度
    返回：
        x_recon: 重构的输入 | mu: 潜在空间的均值 | logvar: 潜在空间的对数方差
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
    
    def forward(self, x):
        z,mu, logvar = self.encoder(x) # 编码器输出的潜在变量的均值以及潜在变量的对数方差。
        x_recon = self.decoder(z) # 解码器从潜在变量z重构出来的图像。
        return x_recon, mu, logvar
    
    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=0.5):
        # 重构损失（BCE：二分类交叉熵）
        BCE = nn.BCELoss(reduction='mean')(recon_x, x)
        # KL散度（KL Divergence）
        KLD = -0.5 *beta  * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  / x.size(0)
        # 总损失
        return BCE + KLD