from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
import torch

# 加载预训练的稳定扩散模型
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipeline = StableDiffusionPipeline.from_pretrained(model_id)
pipeline = pipeline.to(device)

# 定义要生成图像的提示文本
prompt = "tumor MRI"

# 使用模型生成图像
with torch.autocast("cuda"):
    image = pipeline(prompt).images[0]

# 保存生成的图像
image.save("generated_image.png")
print("图像已保存为 generated_image.png")