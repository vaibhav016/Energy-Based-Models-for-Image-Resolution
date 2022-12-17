# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://www.kaggle.com/code/balraj98/single-image-super-resolution-gan-srgan-pytorch

import glob
import zipfile

import wget
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# This custom dataset class has been derived from above two sources.
from utils import bar_progress


class DataDiv2k(Dataset):
    def __init__(self, path, transform=None):
        # self.img_dir = path
        self.transform = transform
        self.image_path = path
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),  # converts a 255 image to 0-1
            ])

        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((256 // 4, 256 // 4)),
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image_hr = Image.open(self.image_path[idx][0])
        image_lr = Image.open(self.image_path[idx][1])

        image_lr = self.lr_transform(image_lr)
        image_hr = self.hr_transform(image_hr)

        return {"lr": image_lr, "hr": image_hr, 'img_name': self.image_path[idx][0].split('/')[-1]}


def build_data_loader(config):
    batch_size = config.ebm_data_config["batch_size"]
    print("---------Download HR images-------")
    wget.download(url=config.ebm_data_config["hr_images"], out=config.ebm_data_config["data_path"], bar=bar_progress)
    print("---------Download LR images-------")
    wget.download(url=config.ebm_data_config["lr_images"], out=config.ebm_data_config["data_path"], bar=bar_progress)
    print("------Download and verification complete-------")

    with zipfile.ZipFile(config.ebm_data_config["data_path"] + "/DIV2K_train_HR", 'r') as zip_ref:
        zip_ref.extractall(config.ebm_data_config["data_path"])

    with zipfile.ZipFile(config.ebm_data_config["data_path"] + "/DIV2K_train_LR_mild", 'r') as zip_ref:
        zip_ref.extractall(config.ebm_data_config["data_path"])

    hr_images = glob.glob(config.ebm_data_config["data_path"] + "/DIV2K_train_HR" + '/*.png')  # returns path of images
    print(len(hr_images))  # contains 800 images

    lr_images = glob.glob(config.ebm_data_config["data_path"] + "/DIV2K_train_LR_mild" + '/*.png')  # returns path of images
    print(len(lr_images))
    train_paths, test_paths = train_test_split(sorted(zip(sorted(hr_images), sorted(lr_images))), test_size=0.02, random_state=42)

    train_dataloader = DataLoader(DataDiv2k(train_paths), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(DataDiv2k(test_paths), batch_size=1)
    return train_dataloader, test_dataloader
