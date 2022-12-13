import glob

from torch.utils.data import DataLoader
import pytorch_lightning as pl
## PyTorch
import torch
# Torchvision
import torchvision

from sklearn.model_selection import train_test_split


## Standard libraries
import os

## Imports for plotting
import matplotlib.pyplot as plt

import matplotlib

from Callbacks import GenerateCallback
from dataset import FacesDataset
from model import CNNModel, DeepEnergyModel
from utils import generate_images_test, get_metrics, compare_imgs

matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()


batch_size = 128
device="cpu"

face_images = glob.glob('/Users/vaibhavsingh/Desktop/NYU/DL project/lfw/**/*.jpg') #returns path of images
print(len(face_images)) #contains 13243 images

train_paths, test_paths = train_test_split(sorted(face_images), test_size=0.2, random_state=42)
train_dataloader = DataLoader(FacesDataset(train_paths), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(FacesDataset(test_paths), batch_size=batch_size)

##############################
sample_image = torch.Tensor(32, 3, 128, 128)
net = CNNModel()
out = net(sample_image)

print(out.shape)
image_samples = next(iter(test_dataloader))
image_sample = image_samples['lr'][0].unsqueeze(1)


def train_model(**kwargs):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=os.path.join("/content/drive/MyDrive/EBM"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=30,
                         gradient_clip_val=0.1)
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = "/Saved_Models/EBM ResNet-18/ResNet_EBM.ckpt"
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = DeepEnergyModel(**kwargs)
        trainer.fit(model, train_dataloader, test_dataloader)
        model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # No testing as we are more interested in other properties
    return model
    '''trainer = pl.Trainer(default_root_dir=os.path.join(os.getcwd(), "../MNIST"),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=5,
                         gradient_clip_val=0.1,
                         # callbacks=[
                         #     GenerateCallback(every_n_epochs=5, batch_size=1),
                         #     SamplerCallback(every_n_epochs=5),
                         #     OutlierCallback(batch_size=1),
                         #
                         # ])
                         )
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = "/epoch=29-step=24330.ckpt"
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = DeepEnergyModel(**kwargs)
        trainer.fit(model, train_dataloader, test_dataloader)
        model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # No testing as we are more interested in other properties
    return model'''

model = train_model(img_shape=(3,256,256),
                    batch_size=train_dataloader.batch_size,
                    lr=1e-4,
                    beta1=0.0)

model.to(device)
pl.seed_everything(43)

callback = GenerateCallback(batch_size=4, vis_steps=8, num_steps=256)
test = next(iter(test_dataloader))

lr_test, hr_test = test['lr'], test['hr']
imgs_per_step = callback.generate_imgs(model, lr_test.to(device))
imgs_per_step = imgs_per_step.cpu()


for i in range(imgs_per_step.shape[1]):
    step_size = callback.num_steps // callback.vis_steps
    imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
    imgs_to_plot = torch.cat([imgs_per_step[0:1,i],imgs_to_plot], dim=0)
    grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1), pad_value=0.5, padding=2)
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(8,8))
    plt.imshow(grid)
    plt.xlabel("Generation iteration")
    plt.xticks([(imgs_per_step.shape[-1]+2)*(0.5+j) for j in range(callback.vis_steps+1)], 
               labels=[1] + list(range(step_size,imgs_per_step.shape[0]+1,step_size)))
    plt.yticks([])
    plt.show()

#Analysis

test_gen_imgs = generate_images_test(test_dataloader, model)
psnr, psnr_lr, ssim, ssim_lr = get_metrics(test_dataloader, test_gen_imgs)

print('Average PSNR - LR images:%.4f and Generated images:%.4f'%(psnr_lr,psnr))
print('Average SSIM - LR images:%.4f and Generated images:%.4f'%(ssim_lr,ssim))

compare_imgs(test_dataloader, test_gen_imgs)

