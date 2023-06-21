#!/usr/bin/env python
# coding: utf-8

# In[2]:


from tqdm.notebook import tqdm, trange

import pandas as pd
import numpy as np
import os

# from contracts import contract

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


from sklearn.model_selection import train_test_split


# In[3]:


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


# In[4]:


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


# In[5]:


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
    predictions = torch.sigmoid(logits) > 0.5

    return predictions


# In[18]:


# Define hyperparameters
LEARNING_RATE = 0.001
MOMENTUM = 0.9
MAX_ITERATIONS = 500
INPUT_SIZE = 29
HIDDEN_SIZES = [100, 100]
OUTPUT_SIZE = 1


# In[19]:


# Initialize the network
net = Net(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

# Define the loss criterion and the training algorithm
# Be careful, use binary cross entropy for binary, CrossEntropy for Multi-class
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)


# In[20]:


path = "/data/mlproject22" if os.path.exists("/data/mlproject22") else "data/"
data = pd.read_csv(os.path.join(path, "transactions.csv.zip"))

X = data.drop(columns=["Time", "Class"])
y = data["Class"]


# In[21]:


x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(
    X, y, test_size=0.25, shuffle=True
)

x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data).reshape(-1, 1)

print(x_train_data.shape)
print(y_train_data.shape)

# In[ ]:


train_neural_network_pytorch(
    net,
    x_train_data,
    y_train_data,
    optimizer,
    criterion,
    iterations=MAX_ITERATIONS,
)

torch.save(net.state_dict(), "model.pth")

# In[ ]:
net = Net(INPUT_SIZE, HIDDEN_SIZES, OUTPUT_SIZE)

net.load_state_dict(torch.load("model.pth"))

# In[ ]:


output = predict_pytorch(net, np.array(x_test_data))


# In[ ]:


wrong_output_count = 0
frauds = 0
frauds_right_detected = 0
frauds_detected = 0
y_test_data = np.array(y_test_data)
for idx, o in enumerate(output):
    if y_test_data[idx] == 1:
        frauds += 1

    if o == y_test_data[idx] and o == 1:
        frauds_right_detected += 1

    if o == 1:
        frauds_detected += 1

    if o != y_test_data[idx]:
        wrong_output_count += 1

print(f"accuracy: {1 - wrong_output_count / len(output)}")

print(f"frauds: {frauds}")
print(f"frauds_detected: {frauds_detected}")
print(f"frauds_right_detected: {frauds_right_detected}")
print(f"frauds detected ratio: {frauds_right_detected / frauds}")

# In[ ]:
