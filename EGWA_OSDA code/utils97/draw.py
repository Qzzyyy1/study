import os
import warnings

import numpy as np
import torch
from typing import List
import matplotlib.pyplot as plt

def getColors():
    return np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255],
            [176, 48, 96], [46, 139, 87], [160, 32, 240], [255, 127, 80], [127, 255, 212],
            [218, 112, 214], [160, 82, 45], [127, 255, 0], [216, 191, 216], [128, 0, 0], [0, 128, 0],
            [0, 0, 128]])

def getClassificationMap(label, unknown=[], unknown_color=[255, 255, 255]):
    colors = getColors()
    image = np.zeros((*label.shape, 3), dtype='uint8')
    for cls in range(1, label.max() + 1):
        image[np.where(label == cls)] = colors[cls - 1]

    for cls in unknown:
        image[np.where(label == cls)] = unknown_color

    return image

def clearBackground(info, image, known_classes=None, unknown_classes=None):

    from utils97.splitData import transformGT

    gt = transformGT(None, info, known_classes, unknown_classes)
    image[np.where(gt == 0)] = [0, 0, 0]

    return image

def parsePredictionLabel(label: List[torch.Tensor], H):

    label = torch.cat(label).cpu() + 1
    return label.reshape(H, -1)

def drawPredictionMap(label: List[torch.Tensor], name, info, known_classes=[], unknown_classes=[], draw_background=True):
    label = parsePredictionLabel(label, info.image_width)
    image = getClassificationMap(label, unknown=[len(known_classes) + 1])
    if draw_background is False:
        image = clearBackground(info, image, known_classes, unknown_classes)
    saveImage(image, name, 'map')

def saveImage(image, name, path='map'):
    if not os.path.exists(path):
        os.makedirs(path)

    plt.imsave(f'{os.path.join(path, name)}.png', image)

def saveFig(filename, path='map', dpi=300):
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(f'{os.path.join(path, filename)}.png', dpi=dpi)
