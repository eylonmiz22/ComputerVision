"""
Implements a network visualization in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
import torchvision.transforms as T
import numpy as np
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from PIL import Image
from a4_helper import *


def hello():
  """
  This is a sample function that we will try to import and run to ensure that
  our environment is correctly set up on Google Colab.
  """
  print('Hello from network_visualization.py!')

def compute_saliency_maps(X, y, model):
  """
  Compute a class saliency map using the model for images X and labels y.

  Input:
  - X: Input images; Tensor of shape (N, 3, H, W)
  - y: Labels for X; LongTensor of shape (N,)
  - model: A pretrained CNN that will be used to compute the saliency map.

  Returns:
  - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
  images.
  """
  # Make input tensor require gradient
 
  ##############################################################################
  # TODO: Implement this function. Perform a forward and backward pass through #
  # the model to compute the gradient of the correct class score with respect  #
  # to each input image. You first want to compute the loss over the correct   #
  # scores (we'll combine losses across a batch by summing), and then compute  #
  # the gradients with a backward pass.                                        #
  # Hint: X.grad.data stores the gradients                                     #

  X.requires_grad_()
  N = X.shape[0]
  score_max = None
  for i in range(N):
      scores = model(X[i][None])
      score_max_index = scores.argmax()
      if i == 0:
          score_max = scores[0,score_max_index]
      else:
          score_max += scores[0,score_max_index]

  score_max.backward()
  saliency, _ = torch.max(X.grad.data.abs(),dim=1)

  return saliency

def make_adversarial_attack(x, y_target, model, max_iter=100, verbose=True):
  """
  Generate an adversarial attack that is close to X, but that the model classifies
  as target_y.

  Inputs:
  - X: Input image; Tensor of shape (1, 3, 224, 224)
  - target_y: An integer in the range [0, 1000)
  - model: A pretrained CNN
  - max_iter: Upper bound on number of iteration to perform
  - verbose: If True, it prints the pogress (you can use this flag for debugging)

  Returns:
  - X_adv: An image that is close to X, but that is classifed as target_y
  by the model.
  """
  # Initialize our adversarial attack to the input image, and make it require gradient
 
  
  ##############################################################################
  # TODO: Generate an adversarial attack X_adv that the model will classify    #
  # as the class target_y. You should perform gradient ascent on the score     #
  # of the target class, stopping when the model is fooled.                    #
  # When computing an update step, first normalize the gradient:               #
  #   dX = learning_rate * g / ||g||_2                                         #
  #                                                                            #
  # You should write a training loop.                                          #
  #                                                                            #
  # HINT: For most examples, you should be able to generate an adversarial     #
  # attack in fewer than 100 iterations of gradient ascent.                    #
  # You can print your progress over iterations to check your algorithm.       #
  ##############################################################################
  loss_fn = nn.CrossEntropyLoss()
  num_steps = 6
  step_size=0.01
  eps=0.3
  clamp=(0,1)
  x_adv = x.clone().detach().requires_grad_(True).to(x.device)
  num_channels = x.shape[1]
  y_target = torch.tensor(y_target).unsqueeze(0).to(x.device)
  for i in range(num_steps):
      _x_adv = x_adv.clone().detach().requires_grad_(True)
      prediction = model(_x_adv)
      print(torch.argmax(prediction))
      loss = loss_fn(prediction, y_target)
      loss.backward()
      with torch.no_grad():
          gradients = _x_adv.grad.sign() * step_size
          x_adv -= gradients
          x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)   
      x_adv = x_adv.clamp(*clamp)
  return x_adv
