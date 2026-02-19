import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_cifar10(batch, workers):

  tr = T.Compose([
    T.RandomCrop(32,padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465),
                (0.2470,0.2435,0.2616))
  ])

  te = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914,0.4822,0.4465),
                (0.2470,0.2435,0.2616))
  ])

  train_ds = torchvision.datasets.CIFAR10(
    "./data", train=True, download=True, transform=tr)

  test_ds = torchvision.datasets.CIFAR10(
    "./data", train=False, download=True, transform=te)

  train_dl = DataLoader(
    train_ds,
    batch_size=batch,
    shuffle=True,
    num_workers=workers,
    pin_memory=torch.cuda.is_available()
  )

  test_dl = DataLoader(
    test_ds,
    batch_size=batch,
    shuffle=False,
    num_workers=workers,
    pin_memory=torch.cuda.is_available()
  )

  return train_dl, test_dl
