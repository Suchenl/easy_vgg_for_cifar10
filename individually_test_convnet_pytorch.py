"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from convnet_pytorch import ConvNet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
import os
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import time

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
SAVE_ALL_MODELS_PARAS_DEFAULT = True
# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.

  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch

  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  predicted_labels = torch.argmax(predictions, dim=1)
  correct_num = torch.eq(predicted_labels, targets).sum().item()
  accuracy = correct_num / targets.size(0)
  # raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy


def test():
    """
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    # #######################
    base_folder_name = f"run_{time.strftime('%Y%m%d%H%M%S')}"
    base_folder_path = os.path.join(r'tests', base_folder_name)
    os.makedirs(base_folder_path, exist_ok=True)

    device = 'cuda:0'
    in_chennel = 3
    num_classes = 10
    load_weights_path = 'pretrained_paras/epoch238-TrainAcc=0.973.pth'
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "test": transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    download_dataset = False if os.path.exists(os.path.join(FLAGS.data_dir, 'cifar-10-python.tar.gz')) else True
    batch_size = FLAGS.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    test_dataset = datasets.CIFAR10(root=FLAGS.data_dir,
                                    train=False,
                                    download=download_dataset,
                                    transform=data_transform['test']
                                    )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=nw
                                              )
    test_num = len(test_dataset)
    print(f'test num:{test_num}')

    model = ConvNet(n_channels=in_chennel, n_classes=num_classes).to(device)

    model_weight_path = load_weights_path
    assert os.path.exists(model_weight_path), "weights file: {} does not exist.".format(model_weight_path)
    weights_dict = torch.load(model_weight_path)
    print(model.load_state_dict(weights_dict, strict=False))

    epoch_num = 1
    epoch = 1
    model.eval()
    test_correct = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_step, data in enumerate(test_bar):
            inputs, labels = data
            predictions = model(inputs.to(device))
            step_accuracy = accuracy(predictions.cpu(), labels)
            test_correct += step_accuracy * inputs.size(0)
            test_bar.desc = "Test epoch[{}/{}] acc:{:.4f}".format(epoch,
                                                                  epoch_num,
                                                                  step_accuracy
                                                                  )
    print(f'Epoch[{epoch}/{epoch_num}] Test Accuracy: {test_correct / test_num:.4f}')
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  test()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--save_all_model_paras', type = str, default = SAVE_ALL_MODELS_PARAS_DEFAULT,
                      help='')
  FLAGS, unparsed = parser.parse_known_args()

  main()