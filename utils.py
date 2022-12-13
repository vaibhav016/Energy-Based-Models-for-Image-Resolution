from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from Callbacks import GenerateCallback
from torch.utils.data import Dataset, DataLoader

def generate_images_test(test_dataloader, model):
  model.to(device)
  pl.seed_everything(43)
  callback = GenerateCallback(batch_size=4, vis_steps=8, num_steps=32)
  gen_imgs = []
  for it in test_dataloader:
    lr_test, hr_test = it['lr'], it['hr']
    imgs_per_step = callback.generate_imgs(model, lr_test.to(device))
    imgs_per_step = imgs_per_step.cpu()
    gen = imgs_per_step[-1]
    gen_imgs.append(gen)

  return gen_imgs


  def get_metrics(test_dataloader, test_gen_imgs):

  avg_psnr = 0
  avg_ssim = 0
  avg_psnr_lr = 0
  avg_ssim_lr = 0
  count = 0
  
  for ind, it in enumerate(test_dataloader):
    lr_test, hr_test = it['lr'], it['hr']
    gen = test_gen_imgs[ind]

    psnr = 0
    ssim = 0
    psnr_lr = 0
    ssim_lr = 0
    
    for i, im in enumerate(hr_test):
      psnr = psnr + peak_signal_noise_ratio(hr_test[i].permute(1,2,0).numpy(),gen[i].permute(1,2,0).numpy())  
      psnr_lr = psnr_lr + peak_signal_noise_ratio(hr_test[i].permute(1,2,0).numpy(),lr_test[i].permute(1,2,0).numpy())
      ssim = ssim + structural_similarity(hr_test[i].permute(1,2,0).numpy(),gen[i].permute(1,2,0).numpy(), multichannel=True)
      ssim_lr = ssim_lr + structural_similarity(hr_test[i].permute(1,2,0).numpy(),lr_test[i].permute(1,2,0).numpy(), multichannel=True)
      count = count + 1

    avg_psnr = avg_psnr + (psnr) 
    avg_psnr_lr = avg_psnr_lr + (psnr_lr) 
    avg_ssim = avg_ssim + (ssim) 
    avg_ssim_lr = avg_ssim_lr + (ssim_lr) 

  return avg_psnr/count, avg_psnr_lr/count, avg_ssim/count, avg_ssim_lr/count 


def compare_imgs(test_dataloader, test_gen_imgs):
    for i, it in enumerate(test_dataloader):
    lr, hr = it['lr'], it['hr']
    gen = test_gen_imgs[i]
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1,3,1)
    plt.title('High Res Image')
    plt.imshow(hr[0].permute(1,2,0).numpy())
    fig.add_subplot(1,3,2)
    plt.title('Low Res Image')
    plt.imshow(lr[0].permute(1,2,0).numpy())
    fig.add_subplot(1,3,3)
    plt.title('EBM ResNet Generated Image')
    plt.imshow(gen[0].permute(1,2,0).numpy())
