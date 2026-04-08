import numpy as np

import torch
from torchvision import transforms

def flip_augmentation(data): # arrays tuple 0:(7, 7, 103) 1=(7, 7)
    horizontal = np.random.random() > 0.5 # True
    vertical = np.random.random() > 0.5 # False
    if horizontal:
        data = np.fliplr(data)
    if vertical:
        data = np.flipud(data)
    return data

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
    return alpha * data + beta * noise

def gaussian_noise(image, mean=0, sigma=0.4):
    image = np.asarray(image, dtype=np.float32)
    max = np.max(image)
    min = np.min(image)
    length = max - min
    image = (image - min) / length
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)
    output = image + noise
    output = output * length + min
    # output = np.clip(output, 0, 1)
    # output = np.uint8(output * length)
    return output

# Input data form：(batchsize, channels, patch_size, patch_size) tensor
def Crop_and_resize_batch(data, HalfWidth):
    da = transforms.RandomResizedCrop(2 * HalfWidth + 1, scale = (0.08, 1.0), ratio=(0.75, 1.3333333333333333))
    x = da(data)
    return x

# Input data form: (patch_size, patch_size, channels) numpy
def Crop_and_resize_single(data, HalfWidth):
    da = transforms.RandomResizedCrop(2 * HalfWidth + 1, scale = (0.08, 1.0), ratio=(0.75, 1.3333333333333333))
    data = data.transpose(2, 0, 1)
    x = da(torch.from_numpy(data))
    x = x.numpy()
    x = x.transpose(1, 2, 0)
    return x

def random_flip(data):
    if torch.rand(1) < 0.5:
        data = torch.flip(data, [-1])
    if torch.rand(1) < 0.5:
        data = torch.flip(data, [-2])
    return data

