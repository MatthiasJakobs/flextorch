import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from flextorch import PytorchClassifier
from seedpy import fixedseed

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(15, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

if __name__ == "__main__":
    device = torch.device("cpu")

    X, y = make_classification(n_samples=10000, n_classes=2, n_features=15, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_val = torch.from_numpy(y_val).long()
    y_test = torch.from_numpy(y_test).long()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    test_ds = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=50)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=50)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=50)

    with fixedseed(torch, 0):
        model = Net()

    config = {"epochs": 500, "seed": 10, "model_persistence_frequency": 50, "early_stopping": {"patience": 50}} 
    container = PytorchClassifier(model, config)
    container.fit(train_loader, val_loader)

    print("First model")
    print(container.train_metrics)
    container.save("classification_results")
    container.plot_results(["ce", "accuracy"], "classification_results/plot.pdf")

    with fixedseed(torch, 0):
        model2 = Net()

    print("After loading")
    container2 = PytorchClassifier.load(model2, "classification_results")
    eval = container2.evaluate_model(test_loader, prefix='test')
    print(eval)
