### easy_vgg_for_cifar10
# During training at epoch 238, achieved a test accuracy of 0.8952, meeting the required threshold of 0.75. Further training may lead to higher accuracy.

# Accuracy and loss curves was saved in 'trains/run_20240716112647' directory.

# convnet_pytorch.py
This script defines the model architecture for CIFAR-10 dataset.

# train_convnet_pytorch.py
This script trains the model for each epoch and evaluates the model when `(epoch % eval_freq == 0)`. It saves the epoch-wise accuracy and loss history.

# train_convnet_pytorch_testByStepFrequency.py
This script trains the model for each epoch, evaluating the model when `(step % eval_freq == 0)`. In addition to epoch-wise history, it also saves step-wise accuracy and loss.

# individually_test_convnet_pytorch.py
This script tests the model using pretrained weights specified by the given path.
