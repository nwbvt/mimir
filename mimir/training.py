
class Trainer(object):

    def __init__(self, data, model, loss_fn, optimizer, device="cpu", seed=0, train_size=0.9):
        generator = None
        if seed:
            generator = torch.Generator().manual_seed(seed)
            torch.manual_seed(seed)
        self.train_data, self.test_data = torch.utils.data.random_split(data, [train_size, 1-train_size],
                                                                        generator=generator)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

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

    def test(self, train_data=False, metrics=[]):
        data = self.train_data if train_data else self.test_data
        size = len(data.dataset)
        num_batches = len(data)
        self.model.eval()
        loss = 0
        metric_results = [0 for _ in metrics]
        with torch.no_grad():
            for x, y in data:
                pred = self.eval(x, y)
                loss = self.loss_fn(pred, y.to(self.device))
                for i, metric in enumerate(metrics):
                    metric_results[i] += metric(x, pred)
        loss /= num_batches
        metric_results = [result/num_batches for result in metric_results]
        return loss, metric_results
        
