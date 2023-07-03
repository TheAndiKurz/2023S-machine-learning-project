# Define hyperparameters
from hyperparameters import *

from tqdm.notebook import trange

import numpy as np

# from contracts import contract

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(Net, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size

        self.hidden_layers = []
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # Follow these steps:
        #
        # Flatten the input x keeping the batch dimension the same
        # Use the relu activation on the output of self.fc1(x)
        # Use the relu activation on the output of self.fc2(x)
        # Pass x through fc3 but do not apply any activation function (think why not?)
        x = x.view(-1, self.input_size)
        x = F.relu(self.input_layer(x))

        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        x = self.output_layer(x)

        return x  # Return x (logits)


def train_neural_network_pytorch(
    net, inputs, labels, optimizer, criterion, iterations=10000
):
    """
    Function for training the PyTorch network.

    :param net: the neural network object
    :param inputs: numpy array of training data values
    :param labels: numpy array of training data labels
    :param optimizer: PyTorch optimizer instance
    :param criterion: PyTorch loss function
    :param iterations: number of training steps
    """
    net.train()  # Before training, set the network to training mode

    # Get the inputs; data is a list of [inputs, labels]
    # Convert to tensors if data is in the form of numpy arrays
    if not torch.is_tensor(inputs):
        inputs = torch.from_numpy(inputs.astype(np.float32))

    if not torch.is_tensor(labels):
        labels = torch.from_numpy(labels.astype(np.float32))

    for iter in trange(iterations):  # loop over the dataset multiple times
        # It is a common practice to track the losses during training
        # Feel free to do so if you want

        # Follow these steps:
        # 1. Reset gradients: Zero the parameter gradients (Check the link for optimizers in the text cell
        #                     above to find the correct function)
        # 2. Forward: Pass `inputs` through the network. This can be done calling
        #             the `forward` function of `net` explicitly but there is an
        #             easier way that is more commonly used
        # 3. Compute the loss: Use `criterion` and pass it the `outputs` and `labels`
        #                      Check the link in the text cell above for details
        # 4. Backward: Call the `backward` function in `loss`
        # 5. Update parameters: This is done using the optimizer's `step` function.
        #                       Check the link provided for details.

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print("Finished Training")


def predict_pytorch(net, X):
    """
    Function for producing network predictions
    """

    net.eval()

    # Make predictions (class 0 or 1) using the learned parameters

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    if not torch.is_tensor(X):
        X = torch.from_numpy(X.astype(np.float32))

    logits = net(X)

    predictions = F.sigmoid(logits) > OUTPUT_THRESHOLD

    return predictions
