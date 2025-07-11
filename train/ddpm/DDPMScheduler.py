import torch

from config.configs import Configs
from train.scheduler import Scheduler


class DDPMScheduler(Scheduler):
    def __init__(self, config:Configs) -> None:
        super().__init__(config)

    def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, noisy_image: torch.Tensor):
        # 计算前一时刻的时间步
        prev_t = self.prev_timestep(timestep)

        alpha_bar_at_t = self.alphas_cumprod[timestep] #ᾱt
        alpha_bar_at_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0) #ᾱ(t-1)

        beta_bar_at_t = 1 - alpha_bar_at_t # 1-ᾱt
        beta_bar_at_prev_t = 1 - alpha_bar_at_prev_t # 1-ᾱ(t-1)

        current_alpha_t = alpha_bar_at_t / alpha_bar_at_prev_t # ᾱt/ᾱ(t-1)
        current_beta_t = 1 - current_alpha_t # 1-ᾱt/ᾱ(t-1)

        # 根据噪声预测 x0 , 去噪后的图像为denoised_image
        # x_0 = ( x_t - √1-α_bar_t ε ) / √α_bar_t
        # x_t = √α_bar_t x_0 + √1-α_bar_t ε
        denoised_image = (noisy_image - torch.sqrt(beta_bar_at_t) * noise_pred) / torch.sqrt(alpha_bar_at_t)

        # 将图像范围限制在 [-1, 1]
        denoised_image = denoised_image.clamp(-self.config.clip, self.config.clip)

        # 根据公式计算均值 ~μ，
        # 这里也可以根据 μ=1/√α_t (x_t - (1-α_t)/√1-α_bar_t ε_t) 得到
        pred_original_sample_coeff = (torch.sqrt(alpha_bar_at_prev_t) * current_beta_t) / beta_bar_at_t
        current_sample_coeff = torch.sqrt(current_alpha_t) * beta_bar_at_prev_t / beta_bar_at_t
        pred_prev_image = pred_original_sample_coeff * denoised_image + current_sample_coeff * noisy_image

        # 加入噪声 σ_t z
        # 其中, σ_t^2 = ( 1-α_bar_(t-1)/1-α_bar_t ) β_t
        variance = 0
        if timestep > 0:
            z = torch.randn(noise_pred.shape).to(device=self.config.device)
            variance = (1 - alpha_bar_at_prev_t) / (1 - alpha_bar_at_t) * current_beta_t
            variance = torch.clamp(variance, min=1e-20)
            variance = torch.sqrt(variance) * z

        return pred_prev_image + variance