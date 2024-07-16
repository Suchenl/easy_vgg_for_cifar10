"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """
  def __init__(self, n_channels, n_classes):

    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu1 = nn.ReLU()
    self.maxpool1 = nn.MaxPool2d(3, stride=2, padding=1)

    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(128)
    self.relu2 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(3, stride=2, padding=1)

    self.conv3_a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn3_a = nn.BatchNorm2d(256)
    self.relu3_a = nn.ReLU()
    self.conv3_b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.bn3_b = nn.BatchNorm2d(256)
    self.relu3_b = nn.ReLU()
    self.maxpool3 = nn.MaxPool2d(3, stride=2, padding=1)

    self.conv4_a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.bn4_a = nn.BatchNorm2d(512)
    self.relu4_a = nn.ReLU()
    self.conv4_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn4_b = nn.BatchNorm2d(512)
    self.relu4_b = nn.ReLU()
    self.maxpool4 = nn.MaxPool2d(3, stride=2, padding=1)

    self.conv5_a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn5_a = nn.BatchNorm2d(512)
    self.relu5_a = nn.ReLU()
    self.conv5_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.bn5_b = nn.BatchNorm2d(512)
    self.relu5_b = nn.ReLU()
    self.maxpool5 = nn.MaxPool2d(3, stride=2, padding=1)

    self.linear = nn.Linear(512, n_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.maxpool1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)
    x = self.maxpool2(x)

    x = self.conv3_a(x)
    x = self.bn3_a(x)
    x = self.relu3_a(x)
    x = self.conv3_b(x)
    x = self.bn3_b(x)
    x = self.relu3_b(x)
    x = self.maxpool3(x)

    x = self.conv4_a(x)
    x = self.bn4_a(x)
    x = self.relu4_a(x)
    x = self.conv4_b(x)
    x = self.bn4_b(x)
    x = self.relu4_b(x)
    x = self.maxpool4(x)

    x = self.conv5_a(x)
    x = self.bn5_a(x)
    x = self.relu5_a(x)
    x = self.conv5_b(x)
    x = self.bn5_b(x)
    x = self.relu5_b(x)
    x = self.maxpool5(x)

    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)

    out = self.linear(x)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
