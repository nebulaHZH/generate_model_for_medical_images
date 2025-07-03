### Scheduler
    扩散模型的核心组件，主要负责噪声的调度和去噪的过程。

#### 主要功能
1. 噪声调度管理
2. 前向过程加噪
3. 反向过程去噪

#### 函数说明
##### set_timesteps
    设置推理的时间步数，生成一个时间序列的数组。
    例如：当 inference step 设为 100, train step 为 1000 时，1000÷100=10，则生成一个 [990, 980, 970, ..., 0] 的数组。
公式：$$t_{i} = \frac{N_{train}-1}{N_{inference}} \times i$$
##### add_noise
    根据采样的时间点，对图像进行加噪。允许从原始图像x₀直接跳转到任意时间步t的噪声图像x_t。
公式：$$x_{t} = \sqrt{\alpha_{t}}x_{0} + \sqrt{1-\alpha_{t}}\epsilon$$

##### prev_timestep
    返回前一个时间步的索引。
公式：$$t_{i-1} = t_{i} - \frac{N_{train}-1}{N_{inference}}$$
