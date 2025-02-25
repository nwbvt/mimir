import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from mimir import training

# example taken from https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())
batch_size = 64
loss_fn = nn.CrossEntropyLoss()
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = training.Trainer(training_data, model, loss_fn, optimizer, device)
trainer.train(100)

