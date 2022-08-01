from email.policy import default
from tkinter import W
import numpy as np
import pandas as pd
import torch
import tqdm
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader
from seedpy import fixedseed
from sklearn.metrics import accuracy_score, f1_score
from os import mkdir, listdir
from os.path import isfile, exists, join
from operator import le, ge
from .utils import deep_dict_update, to_tensor, split_data, to_numpy

def accuracy(predictions, labels):
    if len(predictions.shape) != 1:
        predictions = predictions.argmax(dim=-1)
    return accuracy_score(labels.numpy(), predictions.numpy())


class PyTorchBaseClass:

    def __init__(self, model, config):
        self.model = model
        self.config = {
            "seed": None,
            "device": "cpu",
            "epochs": 100,
            "batch_size": 16,
            "model_persistence_frequency": 10,
            "optimizer": "sgd",
            "optimizer_parameters": {
                "lr": 2e-3,
            },
            "verbose": True,
            "num_workers": 1,
        }
        self.metric_mapping = {
            "nll": F.nll_loss,
            "bce": F.binary_cross_entropy,
            "ce": F.cross_entropy,
            "accuracy": accuracy,
            "mse": F.mse_loss,
        }

        self.optim_mapping = {
            "sgd": optim.SGD,
            "adam": optim.Adam,
        }

        self.config, _ = deep_dict_update(self.config, config)
        self.check_config()

        self.device = self.config["device"]
        self.model.to(self.device)
        self.model_checkpoints = []
        self.rng = np.random.RandomState(self.config['seed'])

    def run_epoch(self, train_loader, optimizer, epoch):
        self.model.train()
        #train_metrics = {f"train_{metric['name']}": [] for metric in self.config["loss_fn"]}
        train_metrics = {}
        for data in train_loader:
            optimizer.zero_grad()
            train_metrics = self.calculate_and_propagate_loss(data, train_metrics)
            optimizer.step()

        agg_metrics = self.aggregate_losses_epoch(train_metrics)
        agg_metrics["epoch"] = epoch
        return agg_metrics

    def convert_to_loader(self, train_data, shuffle=False):
        if isinstance(train_data, torch.utils.data.DataLoader):
            return train_data
        if isinstance(train_data, tuple):
            # Check if tensors or np arrays
            for obj in train_data:
                if not isinstance(obj, (torch.Tensor, np.ndarray)):
                    raise NotImplementedError('If data is given as tuple, it has to be either a tensor or a numpy array but got', type(obj))

            tensors = tuple(to_tensor(obj, self.device) for obj in train_data)

            # Create Dataset and loader
            ds = TensorDataset(*tensors)
            dl = DataLoader(ds, batch_size=self.config['batch_size'], shuffle=shuffle, num_workers=self.config['num_workers'], pin_memory=True)

            return dl
                
        else:
            raise NotImplementedError('Unknown datatype for data', type(train_data))

    def fit(self, train_data, val_data=None):

        # convert data to dataloaders (if needed)
        if val_data is None:
            train_data, val_data = split_data(train_data, percentage_train=0.8, rng=self.rng)
        train_loader = self.convert_to_loader(train_data, shuffle=False)
        val_loader = self.convert_to_loader(val_data, shuffle=True)

        optimizer = self.optim_mapping[self.config["optimizer"]](self.model.parameters(), **self.config["optimizer_parameters"])

        train_log = []

        early_stopping_counter = 0
        best_value = None
        comparison_fn = ge if self.config["early_stopping"]["biggerisbetter"] else le

        with fixedseed(torch, seed=self.config["seed"]):
            with tqdm.tqdm(total=self.config["epochs"]) as progress:
                for epoch in range(self.config["epochs"]):
                    train_metrics = self.run_epoch(train_loader, optimizer, epoch)
                    val_metrics = self.evaluate_model(val_loader)
                    train_metrics.update(val_metrics)
                    train_log.append(train_metrics)

                    self.take_model_snapshot(epoch)

                    self.on_epoch_end()

                    progress_dict = {}
                    progress_dict.update(train_metrics)
                    progress_dict.update(val_metrics)
                    del progress_dict["epoch"]
                    progress.set_postfix(progress_dict)
                    progress.update()

                    # Check for early stopping
                    if early_stopping_counter >= self.config["early_stopping"]["patience"]:
                        #print("Early stopping at epoch", epoch)
                        break

                    current_value = val_metrics[f"val_{self.config['early_stopping']['metric']}"]
                    if epoch == 0:
                        best_value = current_value
                        early_stopping_counter += 1
                        continue

                    if comparison_fn(current_value, best_value):
                        best_value = current_value
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1


        self.take_model_snapshot(self.config["epochs"], end_of_training=True)

        self.train_metrics = train_log 

        self.on_train_end()

    def take_model_snapshot(self, epoch, end_of_training=False):
        if end_of_training or ((epoch+1) % self.config["model_persistence_frequency"] == 0):
            state_dict = self.model.state_dict().copy()
            self.model_checkpoints.append({"epoch": epoch, "state_dict": state_dict})

    def save(self, path, override=True):
        if not exists(path):
            mkdir(path)
        else:
            if not override:
                return

        # Save all checkpoints
        checkpoints_dir = join(path, "model_checkpoints")
        if not exists(checkpoints_dir):
            mkdir(checkpoints_dir)

        for checkpoint in self.model_checkpoints:
            epoch = checkpoint["epoch"]
            state_dict = checkpoint["state_dict"]
            torch.save(state_dict, join(checkpoints_dir, f"checkpoint_{str(epoch).zfill(4)}.pth"))

        # Write hyperparameters
        with open(join(path, "hyperparameters.json"), "w") as fp:
            json.dump(self.config, fp, indent=4)

        # Write train logs + final evaluation
        to_write = {
            "train_log": self.train_metrics,
            #"evaluation": self.eval_metrics
        }
        with open(join(path, "log.json"), "w") as fp:
            json.dump(to_write, fp, indent=4)

    @staticmethod
    def load(model, path):

        with open(join(path, "log.json"), "r") as fp:
            log = json.load(fp)
            train_metrics = log["train_log"]
            #eval_metrics = log["evaluation"]

        with open(join(path, "hyperparameters.json"), "r") as fp:
            config = json.load(fp)

        checkpoint_dir = join(path, "model_checkpoints")
        new_checkpoints = []
        for filename in listdir(checkpoint_dir):
            checkpoint_path = join(checkpoint_dir, filename)
            checkpoint_epoch = int(filename.split('_')[-1].replace(".pth", ""))
            new_checkpoints.append({"epoch": checkpoint_epoch, "state_dict": torch.load(checkpoint_path)})

        model.load_state_dict(new_checkpoints[-1]["state_dict"])
        container = PyTorchBaseClass(model, config)
        container.model_checkpoints = new_checkpoints
        container.train_metrics = train_metrics
        #container.eval_metrics = eval_metrics
        return container

    def on_epoch_end(self):
        pass

    def on_train_end(self):
        pass

    def check_config(self):
        pass

    def calculate_and_propagate_loss(self, data, train_metrics):
        X, y = data
        X = X.to(self.device)
        y = y.to(self.device)
        output = self.model(X)

        for metric in self.config["loss_fn"]:
            loss_name = metric["name"]
            loss_fn = self.metric_mapping[loss_name]

            if self.config["classification"] == False:
                output = output.squeeze()

            loss = loss_fn(output, y)
            loss.backward()

            train_metrics = self.log_metric(train_metrics, f"train_{loss_name}", loss.item())

        return train_metrics

    def aggregate_losses_epoch(self, train_metrics):
        agg_metrics = train_metrics.copy()
        for metric_name, metric_values in train_metrics.items():
            agg_metrics[metric_name] = np.mean(metric_values)
        return agg_metrics

    def evaluate_model(self, data_loader, prefix='val'):
        self.model.eval()
        ys = []
        predictions = []

        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data)
                if self.config["classification"] == False:
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
                val_metrics[f"{prefix}_{metric_name}"] = metric_fn(predictions, ys).item()
            except Exception:
                val_metrics[f"{prefix}_{metric_name}"] = metric_fn(predictions, ys)

        return val_metrics

    def plot_results(self, metrics, path, range=None):
        df = pd.DataFrame(self.train_metrics)
        if range is not None:
            df = df[(df["epoch"] >= range[0]) & (df["epoch"] <= range[1])]
        x = df["epoch"].to_numpy()

        train_ys = []
        train_ys_label = []
        val_ys = []
        val_ys_label = []

        for metric in metrics:
            train_label = f"train_{metric}"
            val_label = f"val_{metric}"
            if train_label in df.columns:
                train_ys.append(df[train_label].to_numpy())
                train_ys_label.append(train_label)
            if val_label in df.columns:
                val_ys.append(df[val_label].to_numpy())
                val_ys_label.append(val_label)

        fig, axs = plt.subplots(1, len(metrics))

        for i, (label, y) in enumerate(zip(train_ys_label, train_ys)):
            try:
                axs[i].plot(x, y, label=label, color=f"C{i}")
            except TypeError:
                axs.plot(x, y, label=label, color=f"C{i}")

        for i, (label, y) in enumerate(zip(val_ys_label, val_ys)):
            try:
                axs[i].plot(x, y, label=label, color=f"C{i}", linestyle="dashed")
            except TypeError:
                axs.plot(x, y, label=label, color=f"C{i}")

        fig.legend()
        fig.tight_layout()
        fig.savefig(path)

    def log_metric(self, dictionary, name, value):
        try:
            dictionary[name].append(value)
        except KeyError:
            dictionary[name] = [value]

        return dictionary

class PytorchRegression(PyTorchBaseClass):

    def __init__(self, model, config):
        default_config = {
            "classification": False,
            "loss_fn": [{"name": "mse"}],
            "evaluation_metrics": [{"name": "mse"}],
            "early_stopping": {
                "metric": "mse",
                "patience": 10,
                "biggerisbetter": False,
            }
        }
        default_config, _ = deep_dict_update(default_config, config)

        super().__init__(model, default_config)

class PytorchClassifier(PyTorchBaseClass):
    def __init__(self, model, config):
        default_config = {
            "classification": True,
            "loss_fn": [{"name": "ce"}],
            "evaluation_metrics": [{"name": "ce"}, {"name": "accuracy"}],
            "early_stopping": {
                "metric": "accuracy",
                "patience": 10,
                "biggerisbetter": True,
            }
        }
        default_config, _ = deep_dict_update(default_config, config)

        super().__init__(model, default_config)
        self.score_fn = f1_score

    def predict(self, X, proba=False):
        if not isinstance(X, (torch.Tensor, np.ndarray)):
            raise NotImplementedError('predict can only take tensors or np arrays, not', type(X))

        X = to_tensor(X, self.device)
        
        with torch.no_grad():
            self.model.eval()
            pred = self.model(X).cpu().numpy()

        if proba:
            return pred
        else:
            return np.argmax(pred, axis=-1)

    def predict_proba(self, X):
        return self.predict(X, proba=True)

    def score(self, X, y):
        y_pred = self.predict(X)
        y = to_numpy(y)

        return self.score_fn(y_pred, y)



