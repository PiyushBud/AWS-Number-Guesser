import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# define a Dataset class
class HandNumDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx]


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def get_data_loaders(batch_size, training_dir):
    logger.info("Get train data loader")
    logger.info(os.listdir(training_dir))
    # read and randomize data
    data = np.loadtxt(training_dir + "/train.txt", delimiter=',')
    perm_idx = np.random.permutation(data.shape[0])
    vali_num = int(data.shape[0] * 0.2)
    vali_idx = perm_idx[:vali_num]
    train_idx = perm_idx[vali_num:]

    # Split into training and validation data
    train_data = data[train_idx]
    vali_data = data[vali_idx]

    # Seperate features and labels
    train_features = train_data[:, 1:].astype(np.float32)
    train_labels = train_data[:, 0].astype(int)
    vali_features = vali_data[:, 1:].astype(np.float32)
    vali_labels = vali_data[:, 0].astype(int)

    train_data = HandNumDataset(train_features, train_labels)
    vali_data = HandNumDataset(vali_features, vali_labels)

    # Create data loaders
    return (DataLoader(train_data, batch_size=batch_size),
            DataLoader(vali_data, batch_size=batch_size))


# The training for the model
def train(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get dataloaders
    train_dataloader, vali_dataloader = get_data_loaders(args.batch_size, args.data)

    model = NeuralNetwork().to(device)

    # loss function and optimizer
    lossFn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)

    epochs = 7
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_data(train_dataloader, model, lossFn, optimizer)
        test(vali_dataloader, model, lossFn)

    save_model(model, args.model_dir)


# Executes a single training step
def train_data(dataloader, model, loss_fn, optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, "
          f"Avg loss: {test_loss:>8f} \n")


def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork()

    # Open existing model
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    with open(os.path.join(model_dir, "model.pth"), "wb") as f:
        torch.save(model.state_dict(), f)


def predict_fn(input_object, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model(torch.tensor(input_object).to(device).float())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args, _ = parser.parse_known_args()

    train(args)


