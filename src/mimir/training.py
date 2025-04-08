import torch
import random
from dataclasses import dataclass
from typing import Type, Callable
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import nn, optim

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

@dataclass
class HyperParameters:
    model_params: map
    optimizer_params: map

def train_epoch(model: nn.Module, data: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer,
                batch_size: int=128, device: str=DEVICE, log: bool=True, collate_fn: Callable=None):
    model.train()
    size = len(data)
    loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)
    for batch, (x, y) in enumerate(loader):
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

def test(model: nn.Module, data: DataLoader, loss_fn: nn.Module, batch_size: int=128,
         device: str=DEVICE, metrics: map={}, collate_fn: Callable=None):
    model.eval()
    metric_values = {name: 0 for name, _ in metrics.items()}
    loss = 0
    size = len(data)
    loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_fn)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            loss += loss_fn(preds, y) * len(x)
            for name, metric in metrics.items():
                metric_values[name] += metric(preds, y) * len(x)
    loss /= size
    metric_values = {name: value/size for name, value in metric_values.items()}
    return loss, metric_values

class Result:
    def __init__(self):
        self.min_loss = None
        self.best_epoch = -1
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
        if self.min_loss is None or loss < self.min_loss:
            self.min_loss = loss
            self.best_epoch = len(self.val_losses) - 1
            return True
        return False

    def best_results(self):
        return self.val_losses[self.best_epoch], self.val_metrics[self.best_epoch]

def _pad_collate(data):
    x,y = zip(*data)
    x = pad_sequence(x, batch_first=True)
    y = torch.stack(y)
    return x, y

def train(data: Dataset, model_class: Type[nn.Module], hyper_params: HyperParameters, loss_fn: nn.Module,
          name: str="model", max_epochs: int=100, max_streak: int=5, optimizer_class: Type[optim.Optimizer]=optim.Adam,
          seed: int=0, device: str=DEVICE, train_size: float=0.9, metrics:map={},
          batch_size:int=128, pad: bool=False, log:bool=True, include_train_results=True):
    fname=f"{name}.pth"
    generator = None
    if seed:
        generator = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    collate_fn = _pad_collate if pad else None
    model = model_class(**hyper_params.model_params).to(device)
    optimizer = optimizer_class(model.parameters(), **hyper_params.optimizer_params)
    train_data, val_data = torch.utils.data.random_split(data, [train_size, 1-train_size], generator)
    results = Result()
    if include_train_results:
        results.add_train_result(*test(model, train_data, loss_fn, batch_size, device, metrics, collate_fn))
    results.add_val_result(*test(model, val_data, loss_fn, batch_size, device, metrics, collate_fn))
    streak = 0 # Number of times the train loss has not improved
    for i in range(max_epochs):
        train_epoch(model, train_data, loss_fn, optimizer, batch_size, device, log, collate_fn)
        if include_train_results:
            results.add_train_result(*test(model, train_data, loss_fn, batch_size, device, metrics, collate_fn))
        improved = results.add_val_result(*test(model, val_data, loss_fn, batch_size, device, metrics, collate_fn))
        if log:
            out = f"Epoch {i:3d}: Loss={results.val_losses[-1]:>3.5f} "
            if include_train_results:
                out += f"val, {results.train_losses[-1]:>3.5f} train "
            for metric in metrics:
                out += f"{metric}={results.val_metrics[-1][metric]:>3.5f} "
                if include_train_results:
                    out += f"val {results.train_metrics[-1][metric]:>3.5f} train "
            print(out)
        if improved:
            streak = 0
            torch.save(model.state_dict(), fname)
        else:
            streak += 1
            if streak >= max_streak:
                break

    model.load_state_dict(torch.load(fname))
    if log:
        loss, metrics = results.best_results()
        out = f"Final results: Loss={loss} "
        for metric in metrics:
            out += f"{metric}={metrics[metric]:>3.5f} "
        print(out)
    return results, model
