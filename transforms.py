import torch
from torchvision import transforms as T

class fishTransform():
    def __init__(self, resize=(256, 256), mean=None, std=None):
        self.resize = resize
        self.mean = mean
        self.std = std

    def __call__(self, img, key="train"):
        data_transform = {
            "train": T.Compose([
                T.Resize(self.resize),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=[-90, 90]),
                T.ToTensor()
            ]),
            "val": T.Compose([
                T.Resize(self.resize),
                T.ToTensor()
            ])
        }
        
        return data_transform[key](img)