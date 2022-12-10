# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html
# Taken from the above link

# This model structure is arbitrary. Basically just applying convolution to reduce the dimension from 128 to 1
# This we need to explore.
import torch
import torch.nn as nn

from Sampler import Sampler
import pytorch_lightning as pl
import torch.optim as optim

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CNNModel(nn.Module):

    def __init__(self, hidden_features=64, out_dim=1, **kwargs):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features

        c_hid3 = hidden_features * 2

        # Series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, c_hid1, kernel_size=8, stride=2),  # [16x16] - Larger padding to get 32x32 image
            nn.MaxPool2d(7, stride=2),
            Swish(),

            nn.Conv2d(c_hid1, c_hid2, kernel_size=5, stride=2),  # [8x8]
            nn.MaxPool2d(6, stride=2),
            Swish(),

            nn.Conv2d(c_hid2, c_hid3, kernel_size=2, stride=1),  # [8x8]

            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=1),  # [8x8]
            Swish(),
            nn.Flatten(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x


class DeepEnergyModel(pl.LightningModule):

    def __init__(self, img_shape, batch_size, alpha=0.1, lr=1e-4, beta1=0.0, **CNN_args):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = CNNModel(**CNN_args)
        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(3, *img_shape)

    def forward(self, x):
        z = self.cnn(x)
        return z

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)  # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        images_dict = batch
        hr_image = images_dict["hr"]
        lr_image = images_dict["lr"]

        small_noise = torch.randn_like(hr_image) * 0.005
        hr_image.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Predict energy score for all images
        inp_imgs = torch.cat([hr_image, lr_image], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        reg_loss = self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss

        # Logging
        self.log('loss', loss)
        self.log('loss_regularization', reg_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', real_out.mean())
        self.log('metrics_avg_fake', fake_out.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model

        images_dict = batch
        hr_image = images_dict["hr"]
        lr_image = images_dict["lr"]

        #         real_imgs, _ = batch
        #         fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([hr_image, lr_image], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log('val_contrastive_divergence', cdiv)
        self.log('val_fake_out', fake_out.mean())
        self.log('val_real_out', real_out.mean())