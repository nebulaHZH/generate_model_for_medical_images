from typing import NamedTuple

class Configs(NamedTuple):
    # 项目名称
    proj_name: str
    # 学习率
    lr: float
    # 批次大小
    batch: int
    # 设备
    device: str
    # 训练周期
    epochs: int
    # 保存周期
    save_period: int
    # 样本周期
    sample_period: int
    # 推理图像数量
    num_inference_images: int
    # 训练时间步数
    num_train_timesteps: int
    # 推理时间步数
    num_inference_timesteps: int
    # beta起始值
    beta_start: float
    # beta结束值
    beta_end: float
    # 图像尺寸
    clip: float
    data_path: str
    image_size: int
    num_classes: int
    num_inference_steps: int
    