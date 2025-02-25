import torch
from torch.utils.data import DataLoader

class Trainer(object):

    def __init__(self, data, model, loss_fn, optimizer, device="cpu", seed=0, train_size=0.9, metrics={}, batch_size=128):
        generator = None
        if seed:
            generator = torch.Generator().manual_seed(seed)
            torch.manual_seed(seed)
        train_data, test_data = torch.utils.data.random_split(data, [train_size, 1-train_size],
                                                              generator=generator)
        self.train_data = DataLoader(train_data, batch_size=batch_size)
        self.test_data = DataLoader(test_data, batch_size=batch_size)
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics

    def eval(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        return self.model(x)

    def train_epoch(self, log=True):
        size = len(self.train_data.dataset)
        self.model.train()
        for batch, (x, y) in enumerate(self.train_data):
            pred = self.eval(x, y)
            loss = self.loss_fn(pred, y.to(self.device))
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch % 100 == 0 and log:
                print(f"loss: {loss:>7f} [{batch * len(x):>6d}/{size:>6d}]", end="\r")

    def test(self, train_data=False):
        data = self.train_data if train_data else self.test_data
        size = len(data.dataset)
        num_batches = len(data)
        self.model.eval()
        loss = 0
        metric_results = {name: 0 for name in self.metrics}
        with torch.no_grad():
            for x, y in data:
                pred = self.eval(x, y)
                loss = self.loss_fn(pred, y.to(self.device))
                if self.metrics:
                    for name, metric in self.metrics.items():
                        metric_results[name] += metric(x, pred)
        loss /= num_batches
        metric_results = [result/num_batches for result in metric_results]
        return loss, metric_results

    def train(self, max_epochs, max_streak=10, log=True):
        best_params = None
        best_loss = None
        streak_len = 0
        for i in range(0, max_epochs):
            if log:
                print(f"Epoch {i}")
            self.train_epoch(log)
            train_loss, train_metrics = self.test(True)
            if log:
                print(f"  Train: Loss={train_loss}", end="")
                if train_metrics:
                    for name, metric in train_metrics.items():
                        print(f" {name}={metric}", end="")
                print("")
            test_loss, test_metrics = self.test(False)
            if log:
                print(f"  Test:  Loss={test_loss}", end="")
                if test_metrics:
                    for name, metric in test_metrics.items():
                        print(f" {name}={metric}", end="")
                print("")
            if (best_loss is None) or (test_loss < best_loss):
                streak_len = 0
                best_loss = test_loss
                best_params = self.model.state_dict()
            else:
                streak_len += 1
                if streak_len >= max_streak:
                    break
        self.model.load_state_dict(best_params)
        return self.model
