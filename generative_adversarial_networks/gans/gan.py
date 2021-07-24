from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
NOISE_DIM = 96

def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device='cpu'):
  """
  Generate a PyTorch Tensor of uniform random noise.

  Input:
  - batch_size: Integer giving the batch size of noise to generate.
  - noise_dim: Integer giving the dimension of noise to generate.
  
  Output:
  - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
    random noise in the range (-1, 1).
  """
  noise = 2 * torch.rand(batch_size, noise_dim) - torch.ones(batch_size, noise_dim)
  return noise



def discriminator():
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = nn.Sequential(
  nn.Linear(784, 256),
  nn.LeakyReLU(0.01),
  nn.Linear(256, 256),
  nn.LeakyReLU(0.01),
  nn.Linear(256, 1))
  return model


def generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = nn.Sequential(
  nn.Linear(noise_dim, 1024),
  nn.ReLU(),
  nn.Linear(1024, 1024),
  nn.ReLU(),
  nn.Linear(1024, 784),
  nn.Tanh())
  return model  

def discriminator_loss(logits_real, logits_fake):
  """
  Computes the discriminator loss described above.
  
  Inputs:
  - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
  """
  ones = np.ones(logits_real.shape)
  zeros = np.zeros(logits_fake.shape)
  ones = torch.from_numpy(ones).float().to(logits_real.device)
  zeros = torch.from_numpy(zeros).float().to(logits_fake.device)

  loss_real = F.binary_cross_entropy_with_logits(logits_real, ones)
  loss_fake = F.binary_cross_entropy_with_logits(logits_fake, zeros)
  return loss_fake + loss_real

def generator_loss(logits_fake):
  """
  Computes the generator loss described above.

  Inputs:
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing the (scalar) loss for the generator.
  """
  ones = np.ones(logits_fake.shape)
  ones = torch.from_numpy(ones).float().to(logits_fake.device)
  loss = F.binary_cross_entropy_with_logits(logits_fake, ones)

  return loss

def get_optimizer(model):
  """
  Construct and return an Adam optimizer for the model with learning rate 1e-3,
  beta1=0.5, and beta2=0.999.
  
  Input:
  - model: A PyTorch model that we want to optimize.
  
  Returns:
  - An Adam optimizer for the model with the desired hyperparameters.
  """
  optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
  return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
  """
  Compute the Least-Squares GAN loss for the discriminator.
  
  Inputs:
  - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  loss = 0
  for Dx, DGz in zip(scores_real, scores_fake):
      loss += 0.5 * (Dx-1)**2 + 0.5 *(DGz**2)
  loss = loss/scores_real.shape[0]
  return loss

def ls_generator_loss(scores_fake):
  """
  Computes the Least-Squares GAN loss for the generator.
  
  Inputs:
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  loss = 0
  for DGz in scores_fake:
      loss += 0.5 * (DGz-1)**2
  loss = loss/scores_fake.shape[0]
  return loss


def build_dc_classifier():
  """
  Build and return a PyTorch nn.Sequential model for the DCGAN discriminator implementing
  the architecture in the notebook.
  """
  model = nn.Sequential(
    nn.Unflatten(1, (1, 28, 28)),
    nn.Conv2d(1, 32, kernel_size=5, stride=1),
    nn.LeakyReLU(0.01),
    nn.MaxPool2d((2, 2), stride=2),
    nn.Conv2d(32, 64, kernel_size=5, stride=1),
    nn.LeakyReLU(0.01),
    nn.MaxPool2d((2, 2), stride=2),
    nn.Flatten(),
    nn.Linear(1024, 1024),
    nn.LeakyReLU(0.01),
    nn.Linear(1024, 1))
  return model

def build_dc_generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the DCGAN generator using
  the architecture described in the notebook.
  """
  model = nn.Sequential(
    nn.Linear(noise_dim, 1024),
    nn.ReLU(),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, 7*7*128),
    nn.ReLU(),
    nn.BatchNorm1d(7*7*128),
    nn.Unflatten(1, (128, 7, 7)),
    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
    nn.Tanh(),
    nn.Flatten())
  return model



