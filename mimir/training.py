import torch
import random
from dataclasses import dataclass
from typing import Type
from torch.utils.data import DataLoader
from torch import nn, optim

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

@dataclass
class HyperParameters:
    model_params: map
    optimizer_params: map

def train_epoch(model: nn.Module, data: DataLoader,
                loss_fn: nn.Module, optimizer: optim.Optimizer,
                device:str, log:bool):
    model.train()
    size = len(data)
    for batch, (x, y) in enumerate(data):
        x = x.to(device)
        y = y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if log and batch % 100 == 0:
            loss = loss.item()
            current = (batch + 1) * len(x)
            print(f"loss: {loss:>7f}, [{current:>6d}/{size:>6d}]", end="\r")

def test(model: nn.Module, data: DataLoader, loss_fn: nn.Module,
         device: str=DEVICE, metrics: map={}):
    model.eval()
    metric_values = {name: 0 for name, _ in metrics.items()}
    loss = 0
    size = len(data)
    with torch.no_grad():
        for x, y in data:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss += loss_fn(preds, y) * len(x)
            for name, metric in metrics.items():
                metric_values[name] += metric(preds, y) * len(x)
    loss /= size
    metric_values = {name: value/size for name, value in metric_values}
    return loss, metric_values

class Result:
    def __init__(self):
        self.min_loss = None
        self.train_losses = []
        self.train_metrics = []
        self.val_losses = []
        self.val_metrics = []

    def _add_result(self, loss:float, metric_results:map, train:bool):
        if train:
            losses = self.train_losses
            metrics = self.train_metrics
        else:
            losses = self.val_losses
            metrics = self.val_metrics
        losses.append(loss)
        metrics.append(metric_results)

    def add_train_result(self, loss:float, metric_results:map):
        self._add_result(loss, metric_results, True)

    def add_val_result(self, loss:float, metric_results:map):
        self._add_result(loss, metric_results, False)
        if self.min_loss is None or loss <= self.min_loss:
            self.min_loss = loss
            return True
        return False

def train(data: DataLoader, model_class: Type[nn.Module], hyper_params: HyperParameters, loss_fn: nn.Module,
          name: str="model", max_epochs: int=100, max_streak: int=5, optimizer_class: Type[optim.Optimizer]=optim.Adam,
          seed: int=0, device: str=DEVICE, train_size: float=0.9, metrics:map={}, batch_size:int=128, log:bool=True):
    fname=f"{name}.pth"
    generator = None
    if seed:
        generator = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    model = model_class(**hyper_params.model_params).to(device)
    optimizer = optimizer_class(model.parameters(), **hyper_params.optimizer_params)
    train_split, val_split = torch.utils.data.random_split(data, [train_size, 1-train_size], generator)
    train_data = DataLoader(train_split, batch_size=batch_size)
    val_data = DataLoader(val_split, batch_size=batch_size)
    results = Result()
    results.add_train_result(*test(model, train_data, loss_fn, device, metrics))
    results.add_val_result(*test(model, val_data, loss_fn, device, metrics))
    streak = 0 # Number of times the train loss has not improved
    for i in range(max_epochs):
        train_epoch(model, train_data, loss_fn, optimizer, device, log)
        results.add_train_result(*test(model, train_data, loss_fn, device, metrics))
        improved = results.add_val_result(*test(model, val_data, loss_fn, device, metrics))
        if log:
            out = f"Epoch {i:3d}: Loss={results.train_losses[-1]:>3.5f} train, {results.val_losses[-1]:>3.5f} val"
            for metric in metrics:
                out += f"{metric}={results.train_metrics[-1][metric]:>3.5f} train, {results.val_metrics[-1][metric]:>3.5f} val"
            print(out)
        if improved:
            streak = 0
            torch.save(model.state_dict(), fname)
        else:
            streak += 1
            if streak >= max_streak:
                break

    model.load_state_dict(torch.load(fname))
    return results, model
