from turtle import Turtle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from seedpy import fixedseed

from flextorch import PytorchClassifier

class Net(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 10)
        self.fc2 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

def _get_output_per_class(output, indices):
    # indices reflects an array of size `output.shape[0]` where each integer represents the class to keep
    if len(indices.shape) == 0:
        indices = indices.reshape(-1)

    assert len(output) == len(indices)
    new_output = []
    for i in range(len(output)):
        class_index = indices[i]
        o_c = output[i][class_index]
        new_output.append(o_c.unsqueeze(0))

    return torch.cat(new_output, axis=0)


def GradientXInput(output, X, labels):
    if output.shape[-1] != 1:
       output = _get_output_per_class(output,labels)

    gradients = torch.autograd.grad(outputs=output, inputs=X, grad_outputs=torch.ones_like(output), create_graph=True)[0] 
    return X * gradients

class PytorchClassifierExtraLoss(PytorchClassifier):

    
    def calculate_and_propagate_loss(self, data, train_metrics):
        X, e, y = data
        X = X.to(self.device)
        y = y.to(self.device)
        e = e.to(self.device)

        X.requires_grad = True
        assert X.requires_grad
        assert X.is_leaf
        output = self.model(X)

        explanation = GradientXInput(output, X, y)
        expl_loss = ((explanation * e)**2).sum()

        loss_obj = self.config['loss_fn'][0]
        loss_name = loss_obj['name']
        loss_fn = self.metric_mapping[loss_name]
        classification_loss = loss_fn(output.squeeze(), y)

        total_loss = 0.001 * expl_loss + classification_loss

        total_loss.backward()

        train_acc = f1_score(np.argmax(output.detach().numpy(), axis=-1), y.numpy())

        train_metrics = self.log_metric(train_metrics, 'train_total_loss', total_loss.item())
        train_metrics = self.log_metric(train_metrics, 'train_expl_loss', expl_loss.item())
        train_metrics = self.log_metric(train_metrics, 'train_classification_loss', classification_loss.item())
        train_metrics = self.log_metric(train_metrics, 'train_f1', train_acc)

        return train_metrics

    def evaluate_model(self, data_loader):
        self.model.eval()
        ys = []
        predictions = []

        with torch.no_grad():
            for data, _, target in data_loader:
                output = self.model(data)
                output = output.squeeze()
                ys.extend(target)
                predictions.append(output)

        ys = torch.from_numpy(np.array(ys))
        predictions = torch.Tensor(torch.cat(predictions, axis=0))

        val_metrics = {}
        for metric in self.config["evaluation_metrics"]:
            metric_name = metric["name"]
            metric_fn = self.metric_mapping[metric_name]

            try:
                val_metrics[f"val_{metric_name}"] = metric_fn(predictions, ys).item()
            except Exception:
                val_metrics[f"val_{metric_name}"] = metric_fn(predictions, ys)

        return val_metrics
        
        
if __name__ == '__main__':
    device = torch.device("cpu")

    num_classes = 2
    num_features = 5

    X, y = make_classification(n_samples=100, n_classes=num_classes, n_features=num_features, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

    with fixedseed(torch, 0):
        model = Net(num_features=num_features, num_classes=num_classes)

    config = {
        'epochs': 5, 
        'optimizer_parameters': {
            'lr': 2e-3
        },
        'loss_fn': [
            {'name': 'ce'}
        ], 
        'evaluation_metrics': [
            {'name': 'ce'}, 
            {'name': 'accuracy'}
        ],
        'seed':0,
    }

    container = PytorchClassifier(model, config)
    container.fit((X_train, y_train))
    print(container.predict(X_test))
    print(container.score(X_test, y_test))

    config = {
        'epochs': 500, 
        'optimizer_parameters': {
            'lr': 2e-3
        },
        'loss_fn': [
            {'name': 'ce'}
        ], 
        'evaluation_metrics': [
            {'name': 'ce'}, 
            {'name': 'accuracy'}
        ],
        'seed':0,
    }

    # Test with extra input data (and extra loss)
    with fixedseed(torch, 0):
        model = Net(num_features=num_features, num_classes=num_classes)
    e_train = np.random.binomial(n=1, p=0.5, size=X_train.shape)
    container = PytorchClassifierExtraLoss(model, config)
    container.fit((X_train, e_train, y_train))
