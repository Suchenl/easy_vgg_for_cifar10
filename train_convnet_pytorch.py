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


def train():
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
    base_folder_path = os.path.join(r'trains', base_folder_name)
    os.makedirs(base_folder_path, exist_ok=True)

    device = 'cuda:0'
    in_chennel = 3
    num_classes = 10
    load_weights_path = ''
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                   transforms.RandomCrop(32, padding=4),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2))
                                   ]),
        "test": transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    download_dataset = False if os.path.exists(os.path.join(FLAGS.data_dir, 'cifar-10-python.tar.gz')) else True
    batch_size = FLAGS.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_dataset = datasets.CIFAR10(root=FLAGS.data_dir,
                                     train=True,
                                     download=download_dataset,
                                     transform=data_transform['train']
                                     )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=nw
                                             )

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
    train_num, test_num = len(train_dataset), len(test_dataset)
    print(f'train num:{train_num}, test num:{test_num}')

    model = ConvNet(n_channels=in_chennel, n_classes=num_classes).to(device)
    if load_weights_path != '':
        model_weight_path = load_weights_path
        assert os.path.exists(model_weight_path), "weights file: {} does not exist.".format(model_weight_path)
        weights_dict = torch.load(model_weight_path)
        print(model.load_state_dict(weights_dict, strict=False))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

    history_epoch_loss_train = []
    history_epoch_acc_train = []
    history_epoch_acc_test = []

    epoch_num = FLAGS.max_steps
    for epoch in range(1, 1 + epoch_num):
        # train
        model.train()
        optimizer.zero_grad()
        history_step_loss_train = []
        history_step_acc_train = []
        history_step_acc_test = []

        train_correct = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            inputs, labels = data
            predictions = model(inputs.to(device))
            step_accuracy = accuracy(predictions.cpu(), labels)
            history_step_acc_train.append(step_accuracy)

            train_correct += step_accuracy * inputs.size(0)

            loss = criterion(predictions, labels.to(device))
            loss.backward()
            optimizer.step()
            history_step_loss_train.append(loss.cpu().item())

            train_bar.desc = "Train epoch[{}/{}] loss:{:.3f} acc:{:.3f}".format(epoch,
                                                                                epoch_num,
                                                                                loss,
                                                                                step_accuracy)
        print(f'Epoch[{epoch}/{epoch_num}] Train Accuracy: {train_correct/train_num:.4f}')

        # plot
        history_epoch_loss_train.append(np.mean(history_step_loss_train))
        history_epoch_acc_train.append(train_correct / train_num)
        plt.figure()
        plt.title(f'Train History Loss')
        plt.plot(range(len(history_epoch_loss_train)), history_epoch_loss_train)
        plt.savefig(os.path.join(base_folder_path, f'Train Loss.png'), dpi=300)
        plt.close()

        plt.figure()
        plt.title(f'Train History Accuracy')
        plt.plot(range(len(history_epoch_acc_train)), history_epoch_acc_train)
        plt.savefig(os.path.join(base_folder_path, f'Train Accuracy.png'), dpi=300)
        plt.close()


        # test
        if epoch % FLAGS.eval_freq == 0:
        # if (epoch + 1) % 1 == 0:
            model.eval()
            test_correct = 0.0
            with torch.no_grad():
                test_bar = tqdm(test_loader, file=sys.stdout)
                for test_step, data in enumerate(test_bar):
                    inputs, labels = data
                    predictions = model(inputs.to(device))
                    step_accuracy = accuracy(predictions.cpu(), labels)
                    history_step_acc_test.append(step_accuracy)
                    test_correct += step_accuracy * inputs.size(0)
                    test_bar.desc = "Test epoch[{}/{}] acc:{:.4f}".format(epoch,
                                                                          epoch_num,
                                                                          step_accuracy
                                                                          )
            print(f'Epoch[{epoch}/{epoch_num}] Test Accuracy: {test_correct / test_num:.4f}')
            # plot
            history_epoch_acc_test.append(test_correct / test_num)
            plt.figure()
            plt.title(f'Test History Accuracy')
            plt.plot(np.array(range(len(history_epoch_acc_test))) * FLAGS.eval_freq, history_epoch_acc_test)
            plt.savefig(os.path.join(base_folder_path, f'Test Accuracy.png'), dpi=300)
            plt.close()

            # save paras
            if FLAGS.save_all_model_paras:
                save_weights_dir = base_folder_path + '/paras'
                os.makedirs(save_weights_dir, exist_ok=True)
                torch.save(model.state_dict(), save_weights_dir + "/epoch{}-TrainAcc={:.3f}-TestAcc={:.3f}.pth".
                           format(epoch,
                                  train_correct / train_num,
                                  test_correct / test_num,
                                  )
                           )
        else:
            # save paras
            if FLAGS.save_all_model_paras:
                save_weights_dir = base_folder_path + '/paras'
                os.makedirs(save_weights_dir, exist_ok=True)
                torch.save(model.state_dict(), save_weights_dir + "/epoch{}-TrainAcc={:.3f}.pth".
                           format(epoch,
                                  train_correct / train_num,
                                  )
                           )
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
  train()

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