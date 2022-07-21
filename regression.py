import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch_skeleton import PytorchRegression
from seedpy import fixedseed

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(15, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    device = torch.device("cpu")

    X, y = make_regression(n_samples=10000, n_features=15, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    test_ds = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=50)

    with fixedseed(torch, 0):
        model = Net()

    container = PytorchRegression(model, {"batch_size": 50, "epochs": 15, "seed": 10, "model_persistence_frequency": 50})
    container.fit((X_train, y_train))

    print("First model")
    print(container.train_metrics)
    container.save("regression_results")
    container.plot_results(["mse"], "regression_results/plot.pdf", range=(1, 10))

    with fixedseed(torch, 0):
        model2 = Net()

    print("After loading")
    container2 = PytorchRegression.load(model2, "regression_results")
    eval = container2.evaluate_model(test_loader, prefix='test')
    print(eval)
