from typing import Optional
import torch

from config.configs import Configs
from model.Unet.DFUnet import DFUNet
from train.ddpm.DDPMScheduler import DDPMScheduler
from utils.plot_image import Plotter

def inference(model:DFUNet, scheduler:DDPMScheduler, images: int, input_channels:int,train_image_size:int,device:torch.device, noise: Optional[torch.Tensor] = None):
    # 选择使用固定的噪声来推理，还是随机的噪声来推理
    if noise is None:
        noisy_sample = torch.randn((images, input_channels, train_image_size,  train_image_size)).to(device)
    else:
        noisy_sample = noise
    i = 0
    print(scheduler.inf_timesteps)
    for t in scheduler.inf_timesteps:
        i = i + 1
        # if i % 100 == 0:
        #     Plotter(noisy_sample, f"t={t.item()}").plot()
        with torch.no_grad():   # 不加入这一行显存会溢出
            noisy_pred = model(noisy_sample, t[None].to(device)).sample
            noisy_sample = scheduler.step(noisy_pred, t, noisy_sample)
    print(noisy_sample.mean())
    noisy_sample = (noisy_sample + 1)/2
    noisy_sample = (noisy_sample * 255).type(torch.uint8)
    print(noisy_sample.max())
    return noisy_sample

def generate_image(model:DFUNet, scheduler:DDPMScheduler) -> torch.Tensor:
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_sample = torch.randn((1, 1, 256, 256)).to(device)
    images = noise_sample
    # 根据时间步逐步去噪
    print(scheduler.inf_timesteps)
    for t in scheduler.inf_timesteps: 
        with torch.no_grad():
            noise_pred = model(noise_sample, t[None].to(device)) # Unet 预测噪声
            noise_sample = scheduler.step(noise_pred.sample,t, noise_sample) # 利用预测的噪声来生成
        # 每100个时间步存一次图像
        if t % 1000 == 0: 
            # Plotter(noise_sample,f"timestep={t}").plot()
            images = torch.concat([images, noise_sample],dim=0)
    Plotter(images,f"timestep={len(scheduler.inf_timesteps)},per 1000").plot()
    # 最终结果归一化处理
    # image = (noise_sample + 1) / 2
    # image = (image * 255).type(torch.uint8)
    return images