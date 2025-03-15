import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from mimir import training

# example taken from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class Model(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, size),
                nn.ReLU(),
                nn.Linear(size, size),
                nn.ReLU(),
                nn.Linear(size, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

if __name__=="__main__":
    training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
    test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())
    batch_size = 64
    loss_fn = nn.CrossEntropyLoss()
    hyper_params = training.HyperParameters(model_params={"size": 512}, optimizer_params={"lr": 1e-3})
    results, model = training.train(data=training_data, model_class=Model, hyper_params=hyper_params,
                                    loss_fn=loss_fn, name="Sample", seed=20250314)
    loss, _ = training.test(data=test_data, model=model, loss_fn=loss_fn)
    print(f"Final loss: {loss:4.4f}")

