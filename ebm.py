import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torchvision

from Callbacks import GenerateCallback
from dataset import build_data_loader
from model import DeepEnergyModel
from utils import generate_images_test, get_metrics, compare_imgs, initialise_configs

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

import argparse


def train_ebm(config, device, load_checkpoint=True):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(default_root_dir=os.path.join(config.ebm_data_config["root_path"]),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=config.ebm_train_config["epochs"],
                         gradient_clip_val=0.1)
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = config.ebm_data_config["root_path"]
    if os.path.isfile(pretrained_filename) and load_checkpoint:
        print("Found pretrained model, loading...")
        model = DeepEnergyModel.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = DeepEnergyModel(config)
        trainer.fit(model, train_dataloader, test_dataloader)
        model = DeepEnergyModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    # No testing as we are more interested in other properties
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Image Super Resolution through EBMs")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_YAML,
        help="The file path of model configuration file",
    )
    parser.add_argument('-tr', '--training', type=int, default=0, help='Enter 1 for training and 0 for inference')
    parser.add_argument('-d', '--data_path', type=str, help='Enter path where you want to save model')
    parser.add_argument('-s', '--saved_model', type=str, help='Enter path of saved model')

    args = parser.parse_args()
    config = initialise_configs(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #### Build Data Loaders ###
    train_dataloader, test_dataloader = build_data_loader(config)

    if args.training == 1:
        model = train_ebm(config=config, device=device, load_checkpoint=True)
    else:
        model = train_ebm(config=config, device=device, load_checkpoint=False)

    model.to(device)
    pl.seed_everything(43)
    callback = GenerateCallback(batch_size=4, vis_steps=8, num_steps=32)
    test = next(iter(test_dataloader))
    lr_test, hr_test = test['lr'], test['hr']
    imgs_per_step = callback.generate_imgs(model, lr_test.to(device))
    imgs_per_step = imgs_per_step.cpu()

    for i in range(imgs_per_step.shape[1]):
        step_size = callback.num_steps // callback.vis_steps
        imgs_to_plot = imgs_per_step[step_size - 1::step_size, i]
        imgs_to_plot = torch.cat([imgs_per_step[0:1, i], imgs_to_plot], dim=0)
        grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1, 1), pad_value=0.5, padding=2)
        grid = grid.permute(1, 2, 0)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid)
        plt.xlabel("Generation iteration")
        plt.xticks([(imgs_per_step.shape[-1] + 2) * (0.5 + j) for j in range(callback.vis_steps + 1)],
                   labels=[1] + list(range(step_size, imgs_per_step.shape[0] + 1, step_size)))
        plt.yticks([])
        plt.show()

        # Analysis

        test_gen_imgs = generate_images_test(test_dataloader, model, device)
        psnr, psnr_lr, ssim, ssim_lr = get_metrics(test_dataloader, test_gen_imgs)

        print('Average PSNR - LR images:%.4f and Generated images:%.4f' % (psnr_lr, psnr))
        print('Average SSIM - LR images:%.4f and Generated images:%.4f' % (ssim_lr, ssim))

        compare_imgs(test_dataloader, test_gen_imgs)
