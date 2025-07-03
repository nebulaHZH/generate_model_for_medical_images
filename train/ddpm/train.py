from dataclasses import dataclass

import torch

from config.configs import Configs
from model.Unet.DFUnet import DFUNet
from .DDPMScheduler import DDPMScheduler
from utils.image_data import ImageData
from utils.plot_image import plot_images


config = Configs(
    data_path="D:\\0-nebula\\dataset\\Havard\\MyDatasets\\CT-MRI\\test\\CT",
    image_size=256,
    num_classes=2,
    batch=4,
    epochs=100,
    lr=1e-4,
    save_period=10,
    proj_name="test",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    sample_period=10,
    clip=1.0,
    num_inference_steps=1000,
    num_inference_images=8,
    num_train_timesteps=1000,
    num_inference_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
)



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn

    # 用于显示进度条，不需要可以去掉
    from tqdm.auto import tqdm

    model = DFUNet(
        config.image_size,
        config.num_classes,
        config.num_train_timesteps
    ).to(config.device)
    scheduler = DDPMScheduler(config)

    # diffusers 里面用的是 AdamW,
    # lr 不能设置的太大
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    training_data = ImageData(config.data_path, config.image_size)
    train_dataloader = DataLoader(training_data, batch_size=config.batch, shuffle=True)
    # 显示原图像和加噪后的图像
    l = [training_data[i].unsqueeze(0).to(config.device) for i in range(config.batch)]  # 1，1，256，256
    test_images = torch.concat(tensors=l,dim=0) # 4,1,256,256
    test_labels = torch.concat(tensors=l,dim=0)
    timesteps = scheduler.sample_timesteps(4)
    noise = torch.randn(test_images.shape).to(config.device)
    noisy_image = scheduler.add_noise(image=test_images, noise=noise, timesteps=timesteps)
    plot_images((test_images / 2 + 0.5).clamp(0, 1), titles=test_labels.detach().tolist(), fig_titles="original image", save_dir=config.proj_name)
    plot_images((noisy_image / 2 + 0.5).clamp(0, 1), titles=test_labels.detach().tolist(), fig_titles="noisy image", save_dir=config.proj_name)

    # # 训练模型
    # for ep in range(config.epochs):
    #     progress_bar = tqdm(total=len(train_dataloader))
    #     model.train()
    #     for image, _ in train_dataloader:
    #         batch = image.shape[0]
    #         timesteps = scheduler.sample_timesteps(batch)
    #         noise = torch.randn(image.shape).to(config.device)
    #         noisy_image = scheduler.add_noise(image=image, noise=noise, timesteps=timesteps)

    #         pred = model(noisy_image, timesteps)[0]
    #         loss = torch.nn.functional.mse_loss(pred, noise)
    #         optimizer.zero_grad()
    #         loss.backward()

    #         # gradient clipping, 用来防止 exploding gradients
    #         nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()

    #         progress_bar.update(1)
    #         logs = {"loss": loss.detach().item(), "ep": ep+1}
    #         progress_bar.set_postfix(**logs)

    #     # 保存模型
    #     if (ep+1) % config.save_period == 0 or (ep+1) == config.epochs:
    #         torch.save({
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'loss': loss,
    #         }, r"checkpoints/model_ep" + str(ep+1))

    #     # 采样一些图片
    #     if (ep+1) % config.sample_period == 0:
    #         model.eval()
    #         labels = torch.randint(0, 9, (config.num_inference_images, 1)).to(config.device)
    #         image = inference(model, scheduler, config.num_inference_images, config, label=labels)
    #         image = (image / 2 + 0.5).clamp(0, 1)
    #         plot_images(image, save_dir=config.proj_name, titles=labels.detach().tolist())
    #         model.train()