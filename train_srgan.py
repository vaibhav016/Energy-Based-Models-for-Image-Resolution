from __future__ import print_function, division
from torch.utils.data import Dataset
import os
import pandas as pd
import random
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
# import pytorch_lightning as pl
import glob 

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter


# Torchvision
import torchvision
#from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.model_selection import train_test_split


## Standard libraries
import os
import json
import math
import numpy as np 
import random

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm
#%matplotlib inline 
# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

from tqdm import tqdm
import time

#Import metrics
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from statistics import mean

#import models

from model import SRResNet, _ResidualConvBlock, _UpsampleBlock, Discriminator
from dataset import FacesDataset

# import wget

#Hyperparameters
#Set Batch Size
train_batch_size = 8

#epochs
epochs = 20

# Optimizer parameter
model_lr = 1e-4
model_betas = (0.9, 0.999)
model_eps = 1e-8
model_weight_decay = 0.0

# Loss weights
pixel_weight = 1.0
content_weight = 1.0
adversarial_weight = 0.001

# Dynamically adjust the learning rate policy [100,000 | 200,000]
lr_scheduler_step_size = epochs // 2
lr_scheduler_gamma = 0.1

# How many iterations to print the training result
train_print_frequency = 100
valid_print_frequency = 1

# Feature extraction layer parameter configuration
feature_model_extractor_node = "features.35"
feature_model_normalize_mean = [0.485, 0.456, 0.406]
feature_model_normalize_std = [0.229, 0.224, 0.225]

#Seed
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#Set Model Name
modelName = 'SRGAN'


#Set paths
highRes = 'data/DIV2K_train_HR/'
lowRes = 'data/DIV2K_train_LR_mild/'
modelPath = 'Saved Models/' + modelName + '/'
outputPath = 'outputs/' + modelName + '/'
resultPath = 'results/' + modelName + '/'


# Store Image Paths
hr_images = glob.glob(highRes + '*.png') #returns path of images
#print(len(hr_images))

lr_images = glob.glob(lowRes + '*.png') #returns path of images
#print(len(lr_images))

#create folders to store experiment results, models, outputs
if not os.path.exists(modelPath):
  os.makedirs(modelPath)

if not os.path.exists(outputPath):
  os.makedirs(outputPath)

if not os.path.exists(resultPath):
  os.makedirs(resultPath)  

#func
def load_dataset():
  #Create Dataloaders
  train_paths, test_paths = train_test_split(sorted(zip(sorted(hr_images),sorted(lr_images))), test_size=0.02, random_state=42)

  train_dataloader = DataLoader(FacesDataset(train_paths), batch_size=train_batch_size, shuffle=True)
  test_dataloader = DataLoader(FacesDataset(test_paths), batch_size=1, shuffle = True)

  return train_dataloader, test_dataloader

#load datasets
train_dataloader, test_dataloader = load_dataset()

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Define class for content loss using VGG 16
class _ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
     """

    def __init__(
            self,
            feature_model_extractor_node: str,
            feature_model_normalize_mean: list,
            feature_model_normalize_std: list
    ) -> None:
        super(_ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_node = feature_model_extractor_node
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract the thirty-sixth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data. 
        # This is the VGG model preprocessing method of the ImageNet dataset.
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: Tensor, gt_tensor: Tensor) -> Tensor:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        gt_feature = self.feature_extractor(gt_tensor)[self.feature_model_extractor_node]

        # Find the feature map difference between the two images
        loss = F.mse_loss(sr_feature, gt_feature)

        return loss


# function to build model 
def build_model():
  #Generator
  g_model = SRResNet(upscale_factor=1, 
                     in_channels = 3, 
                     out_channels = 3, 
                     channels = 64, 
                     num_rcb = 16).to(device=device)
  
  #Discriminator
  d_model = Discriminator().to(device=device)

  return d_model, g_model


#function to initialize losses for generator and discriminator
def define_loss(): 
    pixel_criterion = nn.MSELoss()
    content_criterion = _ContentLoss(feature_model_extractor_node=feature_model_extractor_node,
                                           feature_model_normalize_mean=feature_model_normalize_mean,
                                           feature_model_normalize_std=feature_model_normalize_std)
    adversarial_criterion = nn.BCEWithLogitsLoss()

    # Transfer to CUDA
    pixel_criterion = pixel_criterion.to(device)
    content_criterion = content_criterion.to(device)
    adversarial_criterion = adversarial_criterion.to(device)

    return pixel_criterion, content_criterion, adversarial_criterion


#function to initialize optimizers for generator and discriminator
def define_optimizer(d_model, g_model):
    d_optimizer = optim.Adam(d_model.parameters(),
                             model_lr,
                             model_betas,
                             model_eps,
                             model_weight_decay)
    g_optimizer = optim.Adam(g_model.parameters(),
                             model_lr,
                             model_betas,
                             model_eps,
                             model_weight_decay)
    
    return d_optimizer, g_optimizer


#function to initialize schedulers for generator and discriminator
def define_scheduler(d_optimizer, g_optimizer):
    d_scheduler = lr_scheduler.StepLR(d_optimizer,
                                      lr_scheduler_step_size,
                                      lr_scheduler_gamma)
    g_scheduler = lr_scheduler.StepLR(g_optimizer,
                                      lr_scheduler_step_size,
                                      lr_scheduler_gamma)
    return d_scheduler, g_scheduler

# Function for one step of training
def training_step(
        d_model,
        g_model,
        train_dataloader,
        pixel_criterion,
        content_criterion,
        adversarial_criterion,
        d_optimizer,
        g_optimizer,
        psnr_model,
        ssim_model,
        epoch,
        writer
    ):
    
    
    # Calculate how many batches of data are in each Epoch
    num_batches = len(train_dataloader)
    
    #initialize performance metrics lists
    d_losses = []
    g_losses = []
    pixel_losses = [] 
    content_losses = []
    adversarial_losses = []
    d_hr_probabilities = []
    d_fk_probabilities = []
    psnrs = []
    ssims = []

    # Put the adversarial network model in training mode
    d_model.train()
    g_model.train()

    # initialize dataloader as an iterator
    dataloader_iter = iter(train_dataloader)

    for i in tqdm(range(num_batches)):

        #get batch
        batch_data = next(dataloader_iter)

        # Transfer in-memory data to CUDA devices to speed up training
        hr = batch_data["hr"].to(device=device)
        lr = batch_data["lr"].to(device=device)

        # Set the real sample label to 1, and the false sample label to 0
        batch_size, _, height, width = hr.shape
        real_label = torch.full([batch_size, 1], 1.0, dtype=hr.dtype, device=device)
        fake_label = torch.full([batch_size, 1], 0.0, dtype=hr.dtype, device=device)

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        hr_output = d_model(hr)
        d_loss_hr = adversarial_criterion(hr_output, real_label)
        # Call the gradient scaling function in the mixed precision API to
        # back-propagate the gradient information of the fake samples
        d_loss_hr.backward(retain_graph=True)

        # Calculate the classification score of the discriminator model for fake samples
        # Use the generator model to generate fake samples
        fk = g_model(lr)
        fk_output = d_model(fk.detach().clone())
        d_loss_fk = adversarial_criterion(fk_output, fake_label)
        # Call the gradient scaling function in the mixed precision API to
        # back-propagate the gradient information of the fake samples
        d_loss_fk.backward()

        # Calculate the total discriminator loss value
        d_loss = d_loss_hr + d_loss_fk

        # Improve the discriminator model's ability to classify real and fake samples
        d_optimizer.step()
        # Finish training the discriminator model

        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        g_model.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        pixel_loss = pixel_weight * pixel_criterion(fk, hr)
        content_loss = content_weight * content_criterion(fk, hr)
        adversarial_loss = adversarial_weight * adversarial_criterion(d_model(fk), real_label)

        # Calculate psnr and ssim
        psnr_train = psnr_model(hr.permute(0, 2, 3, 1).detach().cpu().numpy(), fk.permute(0, 2, 3, 1).detach().cpu().numpy())
        ssim_train = ssim_model(hr.permute(0, 2, 3, 1).detach().cpu().numpy(), fk.permute(0, 2, 3, 1).detach().cpu().numpy(), multichannel = True)

        # Calculate the generator total loss value
        g_loss = pixel_loss + content_loss + adversarial_loss
        # Call the gradient scaling function in the mixed precision API to
        # back-propagate the gradient information of the fake samples
        g_loss.backward()

        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        g_optimizer.step()
        # Finish training the generator model

        # Calculate the score of the discriminator on real samples and fake samples,
        # the score of real samples is close to 1, and the score of fake samples is close to 0
        d_hr_probability = torch.sigmoid_(torch.mean(hr_output.detach()))
        d_fk_probability = torch.sigmoid_(torch.mean(fk_output.detach()))

        #print(d_hr_probabilities)

        # Statistical accuracy and loss value for terminal data output
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
        pixel_losses.append(pixel_loss.item()) 
        content_losses.append(content_loss.item())
        adversarial_losses.append(adversarial_loss.item()) 
        d_hr_probabilities.append(d_hr_probability.item()) 
        d_fk_probabilities.append(d_fk_probability.item()) 
        psnrs.append(psnr_train)
        ssims.append(ssim_train)


        # Write the data during training to the training log file
        if i % train_print_frequency == 0:
            iters = i + epoch * num_batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(HR)_Probability", d_hr_probability.item(), iters)
            writer.add_scalar("Train/D(FK)_Probability", d_fk_probability.item(), iters)
            writer.add_scalar("Train/PSNR", psnr_train, iters)
            writer.add_scalar("Train/SSIM", ssim_train, iters)
            #progress.display(batch_index + 1)

        del hr, lr, fk, d_loss, g_loss, pixel_loss, content_loss, adversarial_loss, d_hr_probability, d_fk_probability, psnr_train, ssim_train
        torch.cuda.empty_cache()
    
    return (mean(d_losses),
           mean(g_losses), 
           mean(pixel_losses), 
           mean(content_losses), 
           mean(adversarial_losses), 
           mean(d_hr_probabilities), 
           mean(d_fk_probabilities),
           mean(psnrs),
           mean(ssims))


# Function for one step of validation
def validation_step(
        g_model, 
        valid_dataloader,
        epoch,
        writer,
        psnr_model,
        ssim_model,
        mode
    ):
    # Initialize lists for storing metrics
    psnres = []
    ssimes = []

    # Put the adversarial network model in validation mode
    g_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # get length of dataloader
    n = len(valid_dataloader)

    # initialize dataloader as an iterator
    dataloader_iter = iter(valid_dataloader)

    with torch.no_grad():
        for i in tqdm(range(n)):
        # while batch_data is not None:
            
            #get batch
            batch_data = next(dataloader_iter)
            
            # Transfer the in-memory data to the CUDA device to speed up the test
            hr = batch_data["hr"].to(device)
            lr = batch_data["lr"].to(device)

            # Use the generator model to generate a fake sample
            fk = g_model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(hr[0].permute(1, 2, 0).cpu().numpy(), fk[0].permute(1, 2, 0).cpu().numpy())
            ssim = ssim_model(hr[0].permute(1, 2, 0).cpu().numpy(), fk[0].permute(1, 2, 0).cpu().numpy(), multichannel = True)
            psnres.append(psnr) 
            ssimes.append(ssim)

            del hr, lr, fk, psnr, ssim
            torch.cuda.empty_cache()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", mean(psnres), epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", mean(ssimes), epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return mean(psnres), mean(ssimes)


# Function to save performance summary 
def save_train_summary(epochs, train_d_losses, train_g_losses, train_pixel_losses, train_content_losses, 
                       train_adversarial_losses, train_d_hr_probabilities, train_d_fk_probabilities, train_psnrs,
                       train_ssims, val_psnrs, val_ssims):

  epoch_range = [i for i in range(epochs)]

  df = pd.DataFrame(list(zip(epoch_range,
                             train_d_losses,
                             train_g_losses,
                             train_pixel_losses,
                             train_content_losses,
                             train_adversarial_losses,
                             train_d_hr_probabilities,
                             train_d_fk_probabilities,
                             train_psnrs,
                             train_ssims,
                             val_psnrs,
                             val_ssims)),
                    columns = ['epoch', 'train_d_loss', 'train_g_loss', 'train_pixel_loss', 'train_content_loss', 'train_adversarial_loss',
                               'train_d_hr_probability', 'train_d_fk_probability', 'train_psnr', 'train_ssim', 'val_psnr', 'val_ssim'])
  
  df.to_csv(resultPath + 'TrainSummary_SRGAN.csv')



# Function to define training loop
def train_model(epochs = 10):

  # initialize training to generate network evaluation indicators
  best_psnr = 0.0
  best_ssim = 0.0

  #Initiate lists for performance summary
  train_d_losses = [] 
  train_g_losses = []
  train_pixel_losses = [] 
  train_content_losses = []
  train_adversarial_losses = [] 
  train_d_hr_probabilities = []
  train_d_fk_probabilities = []
  train_psnrs = []
  train_ssims = []
  val_psnrs = []
  val_ssims = []

  # load Model
  d_model, g_model = build_model()

  # load Dataset
  train_dataloader, test_dataloader = load_dataset()

  # load loss functions
  pixel_criterion, content_criterion, adversarial_criterion = define_loss()

  # load optimizers
  d_optimizer, g_optimizer = define_optimizer(d_model, g_model)

  # load schedulers
  d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)

  # check if pre-trained model already exists, if yes then load
  if os.path.exists(modelPath + 'best_d_model_srgan.pkl'):
    d_model.load_state_dict(torch.load(modelPath + 'best_d_model_srgan.pkl')['model_state_dict'])
  
  if os.path.exists(modelPath + 'best_g_model_srgan.pkl'):
    g_model.load_state_dict(torch.load(modelPath + 'best_g_model_srgan.pkl')['model_state_dict'])


  # Create training process log file
  writer = SummaryWriter(resultPath + 'SRGAN_summary')

  # note train loop start time
  train_loop_start_time = time.time()

  for epoch in range(epochs):

      print('\nEpoch: ', epoch, '.....')

      train_start_time = time.time()

      print('Training....')
      # take a training step
      (d_loss, 
      g_loss, 
      pixel_loss, 
      content_loss, 
      adversarial_loss, 
      d_hr_probability, 
      d_fk_probability,
      psnr_train,
      ssim_train) = training_step(d_model,
                                        g_model,
                                        train_dataloader,
                                        pixel_criterion,
                                        content_criterion,
                                        adversarial_criterion,
                                        d_optimizer,
                                        g_optimizer,
                                        peak_signal_noise_ratio,
                                        structural_similarity,
                                        epoch,
                                        writer)

      #add to performance summary lists
      train_d_losses.append(d_loss) 
      train_g_losses.append(g_loss) 
      train_pixel_losses.append(pixel_loss) 
      train_content_losses.append(content_loss) 
      train_adversarial_losses.append(adversarial_loss)
      train_d_hr_probabilities.append(d_hr_probability)
      train_d_fk_probabilities.append(d_fk_probability)
      train_psnrs.append(psnr_train)
      train_ssims.append(ssim_train)
      
      #note training end time and val start time
      train_end_time = time.time()
      
      #Print Training time
      print('Training completed in ', float(train_end_time - train_start_time)/60.0, ' mins')

      print('Validation....')
      # take a validation step
      psnr_val, ssim_val = validation_step(g_model,
                                           test_dataloader,
                                           epoch,
                                           writer,
                                           peak_signal_noise_ratio, 
                                           structural_similarity, 
                                           'Valid')

      #add to performance summary lists
      val_psnrs.append(psnr_val)
      val_ssims.append(ssim_val)

      #note val end time
      val_end_time = time.time()

      #Print validation time
      print('Validation completed in ', float(val_end_time - train_end_time)/60.0, ' mins')

      #Print Validation Performance
      print('Epoch: ', epoch, '---> Validation PSNR: ', psnr_val)
      print('Epoch: ', epoch, '---> Validation SSIM: ', ssim_val)

      # Update LR
      d_scheduler.step()
      g_scheduler.step()


      # if avg psnr/ssim improved, then save model
      if psnr_val > best_psnr or ssim_val > best_ssim:
        best_psnr = max(psnr_val, best_psnr)
        best_ssim = max(ssim_val, best_ssim)

        print('Saving model ...')
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': d_model.state_dict(),
                    'optimizer_state_dict': d_optimizer.state_dict(),
                    'scheduler': d_scheduler,
                    'd_loss': d_loss, 
                    'pixel_loss': pixel_loss, 
                    'content_loss': content_loss, 
                    'adversarial_losses': adversarial_loss, 
                    'd_hr_probability': d_hr_probability, 
                    'd_fk_probability': d_fk_probability
                  }
                  , modelPath + 'best_d_model_srgan.pkl')
        
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': g_model.state_dict(),
                    'optimizer_state_dict': g_optimizer.state_dict(), 
                    'scheduler':g_scheduler,
                    'g_loss': g_loss, 
                    'pixel_loss': pixel_loss, 
                    'content_loss': content_loss, 
                    'adversarial_losses': adversarial_loss, 
                    'd_hr_probability': d_hr_probability, 
                    'd_fk_probability': d_fk_probability
                  }
                  , modelPath + 'best_g_model_srgan.pkl')
    

  #Save Performance metrics
  save_train_summary(epochs,
                     train_d_losses,
                     train_g_losses,
                     train_pixel_losses,
                     train_content_losses,
                     train_adversarial_losses,
                     train_d_hr_probabilities,
                     train_d_fk_probabilities,
                     train_psnrs,
                     train_ssims,
                     val_psnrs,
                     val_ssims)


  #Testing

  print('Testing....')
  
  #note test start time
  test_start_time = time.time()

  #Load Best Model
  g_model.load_state_dict(torch.load(modelPath + 'best_g_model_srgan.pkl')['model_state_dict'])

  #run testing loop
  psnr_testing, ssim_testing = validation_step(g_model,
                                               test_dataloader,
                                               epoch,
                                               writer,
                                               peak_signal_noise_ratio, 
                                               structural_similarity, 
                                               'Test')  
  
  #note test end time
  test_end_time = time.time()
  
  #Print testing time and performance
  print('Testing completed in ', float(test_end_time - test_start_time)/60.0, ' mins')
  print('\nTest PSNR: ', psnr_testing)
  print('Test SSIM: ', ssim_testing)

  # Note and Print total loop time
  train_loop_end_time = time.time()
  print('\nTotal Model Training Time: ', float(train_loop_end_time - train_loop_start_time)/60.0, ' mins')



#Train the model for specified epochs
train_model(epochs)

#Function to plot and save lr, hr and generated image
def compare_imgs(test_dataloader, model):
  for i, it in enumerate(test_dataloader):
    lr, hr = it['lr'].to(device), it['hr'].to(device)
    gen = model(lr).detach()
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle('Image ' + it['img_name'][0], x = 0.5, y = 0.75)
    fig.add_subplot(1,3,1)
    plt.title('High Res')
    plt.imshow(hr[0].permute(1,2,0).cpu().numpy())
    fig.add_subplot(1,3,2)
    plt.title('Low Res')
    plt.imshow(lr[0].permute(1,2,0).cpu().numpy())
    fig.add_subplot(1,3,3)
    plt.title(modelName + ' Generated Image')
    plt.imshow(gen[0].permute(1,2,0).cpu().numpy())
    
    #Save comparison image
    plt.savefig(outputPath + 'comparison_' + modelName + '_' + it['img_name'][0])


#make static test dataloader
_, test_paths = train_test_split(sorted(zip(sorted(hr_images),sorted(lr_images))), test_size=0.02, random_state=42)
test_dataloader = DataLoader(FacesDataset(test_paths), batch_size=1)

#build model
_, g_model = build_model()

#load the best model parameters
g_model.load_state_dict(torch.load(modelPath + 'best_g_model_srgan.pkl')['model_state_dict'])

#generate and compare images
compare_imgs(test_dataloader, g_model)
