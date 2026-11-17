

import matplotlib.pyplot as plt
import torch


def addNoise(x, snr):
    # 
    if isinstance(x, torch.Tensor):
        #  PyTorch ， PyTorch 
        x_power = torch.sum(x ** 2) / 1024
        noise_power = x_power / (10 ** (snr / 10.0))
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        noisy_x = x + noise
    elif isinstance(x, np.ndarray):
        #  NumPy ， NumPy 
        x_power = np.sum(x ** 2) / 1024
        noise_power = x_power / (10 ** (snr / 10.0))
        noise = np.random.randn(*x.shape) * np.sqrt(noise_power)
        noisy_x = x + noise
    else:
        raise TypeError("Input x must be either a torch.Tensor or a numpy.ndarray")

    return noisy_x


def addMixedNoise(x, snr, salt_pepper_prob=0.15):
    """
    向信号中添加高斯噪声和椒盐噪声。

    参数:
    - x: 原始信号
    - snr: 信噪比 (信号功率 / 噪声功率的比值，单位：dB)
    - salt_pepper_prob: 椒盐噪声的比例 (默认5%)

    返回:
    - noisy_x: 添加噪声后的信号
    """
    # 
    n = len(x)

    # 
    x_power = np.sum(x ** 2) / n

    # 
    noise_power_total = x_power / (10 ** (snr / 10.0))

    # （）
    gaussian_noise_power = noise_power_total

    # 
    gaussian_noise = np.random.randn(n) * np.sqrt(gaussian_noise_power)

    # 
    noisy_x = x + gaussian_noise  # 
    num_salt_pepper = int(salt_pepper_prob * n)  # 

    # ，
    pepper_indices = np.random.choice(n, num_salt_pepper // 2, replace=False)
    noisy_x[pepper_indices] = np.min(x)  # （“”）

    # ，
    salt_indices = np.random.choice(n, num_salt_pepper // 2, replace=False)
    noisy_x[salt_indices] = np.max(x)  # （“”）

    return noisy_x



def addNoiseBatch(x, snr):
    noisy = []
    for i in range(len(x)):
        new_one = addNoise(x[i], snr)
        noisy.append(new_one)

    if isinstance(x, torch.Tensor):
        noisy = torch.stack(noisy, dim=0)
    else:
        noisy = np.stack(noisy, axis=0)
    return noisy


##############################################################################
# 
def addPinkNoise(data, snr):
    # 
    data_power = np.sum(data ** 2) / len(data)

    # 
    noise_power = data_power / (10 ** (snr / 10.0))

    # 
    pink_noise = np.random.randn(len(data))
    pink_noise = np.cumsum(pink_noise)
    pink_noise -= np.mean(pink_noise)
    pink_noise *= np.sqrt(noise_power / np.var(pink_noise))

    # 
    noisy_data = data + pink_noise

    return noisy_data


def addPinkNoiseBatch(data, snr):
    noisy_data = []
    for i in range(len(data)):
        new_one = addPinkNoise(data[i], snr)
        noisy_data.append(new_one)

    noisy_data = np.stack(noisy_data, axis=0)
    return noisy_data


##################################################################################
# 

import numpy as np


def addLaplaceNoise(x, snr):
    # 
    x_power = np.sum(x ** 2) / 1024

    # 
    noise_power = x_power / (10 ** (snr / 10.0))

    # 
    noise = np.random.laplace(0, np.sqrt(noise_power), 1024)

    # 
    noisy_x = x + noise

    return noisy_x


def addLaplaceNoiseBatch(x, snr):
    noisy = []
    for i in range(len(x)):
        new_one = addLaplaceNoise(x[i], snr)
        noisy.append(new_one)

    noisy = np.stack(noisy, axis=0)
    return noisy


def addNoiseDisplay(sampleOrg, sampleafter):
    org_x = [x for x in sampleOrg]
    after_x = [x for x in sampleafter]

    ## 6.
    t = np.arange(0, len(org_x), 1)  # 01023
    fig, (fig2, fig3) = plt.subplots(1, 2, figsize=(12, 4))

    fig2.plot(t, list(after_x), 'y', label='Original data add noise')
    fig2.legend()
    fig2.set_title('Rank is 1')
    fig2.set_xlabel('Sampling point', size=15)
    fig2.set_ylabel('Value of data', size=15)

    fig3.plot(t, list(org_x), 'b', label='Original data')
    fig3.legend()
    fig3.set_title('Rank is 1')
    fig3.set_xlabel('Sampling point', size=15)
    fig3.set_ylabel('Value of data', size=15)

    plt.show()


if __name__ == '__main__':
    length = 784
    number = 300  # 
    normal = True  # 
    rate = [0.5, 0.25, 0.25]  # 
