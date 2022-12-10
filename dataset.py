# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# This custom dataset class has been derived from above two sources.
class FacesDataset(Dataset):
    def __init__(self, path, transform=None):
        # self.img_dir = path
        self.transform = transform
        self.image_path_list = path
        self.hr_height = 128
        self.hr_width = 128

        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((self.hr_height, self.hr_height), Image.BICUBIC),
                transforms.ToTensor(),  # converts a 255 image to 0-1
            ])

        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((self.hr_height // 4, self.hr_height // 4), Image.BICUBIC),
                transforms.Resize((self.hr_height, self.hr_height), Image.BICUBIC),
                transforms.ToTensor()

            ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_path_list[idx])
        image_lr = self.lr_transform(image)
        image_hr = self.hr_transform(image)

        return {"lr": image_lr, "hr": image_hr}



