import numpy as np 
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

# Define class for generator and discriminator 
# https://arxiv.org/pdf/1511.06434.pdf 
# The paper about deep convolutional generative adversarial networks (DCGAN) give 
# advice about making DCGANs. It is basically the same template as here. 
class Generator(nn.Module):
    def __init__(self, input_size, n_filters):
        super().__init__()

        self.network = nn.Sequential(
            nn.ConvTranspose2d(input_size, n_filters, 3, 1, 0, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_filters, n_filters // 2, 3, 2, 0, bias=False),
            nn.BatchNorm2d(n_filters // 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_filters // 2, n_filters // 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters // 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(n_filters // 4, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        'Forward pass'
        output = self.network(x)
        return output
    
class Discriminator(nn.Module):
    def __init__(self, n_filters):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters // 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters // 4, n_filters // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters // 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters // 2, n_filters, 3, 2, 0, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_filters, 1, 3, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        'Forward pass'
        output = self.network(x)
        return output.view(-1, 1).squeeze(0) 
    

# Generate noise from normal distribution 
def create_noise(batch_size, z_size):
    input_z = torch.normal(0.0, 1.0, (batch_size, z_size, 1, 1))
    return input_z

# The loss function is for GAN 
def gan_discriminator_loss(real_output, fake_output):
  """Added negative sign to minimize instead of maxmize"""
  num_samples = real_output.shape[0]
  loss = -(1 / num_samples) * torch.sum( torch.log(real_output) + (torch.log(1.0 - fake_output)))
  return loss

def gan_generator_loss(fake_output):
  """Added negative sign to minimize instead of maxmize"""
  num_fake_samples = fake_output.shape[0]
  loss = -(1 / num_fake_samples) * torch.sum((torch.log(fake_output)))
  return loss 

def value_func(real_output, fake_output):
   num_fake_samples = fake_output.shape[0]
   num_real_samples = real_output.shape[0]
   dis_loss_real = -(1 / num_fake_samples) * torch.sum(torch.log(real_output)).item()
   dis_loss_fake = -(1 / num_real_samples) * torch.sum(torch.log(1.0 - fake_output)).item()
   return dis_loss_real, dis_loss_fake

# The loss function is for WGFAN 
def wgan_discriminator_loss(real_output, fake_output):
  '''Added negative sign to minimize instead of maximize'''
  loss = -(real_output.mean() - fake_output.mean())
  return loss

def wgan_generator_loss(fake_output):
  loss = -fake_output.mean()
  return loss 