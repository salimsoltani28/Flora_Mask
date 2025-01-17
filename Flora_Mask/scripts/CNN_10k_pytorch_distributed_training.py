import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from tqdm import tqdm
import logging
import copy
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Paths and constants
checkpoint_path = "/path/to/your/checkpoints/directory"
data_path = "/path/to/your/data"
num_img_per_class = 4000
batch_size = 16
num_epochs = 150
num_classes = 10

class CustomBalancedSampler(DistributedSampler):
    """A custom sampler that oversamples classes with fewer than threshold samples."""
    def __init__(self, dataset, num_replicas=None, rank=None, threshold=num_img_per_class, shuffle=True):
        super().__init__(dataset, num_replicas, rank, shuffle)
        _, class_counts = np.unique(np.array(dataset.targets), return_counts=True)
        self.weights = torch.tensor([threshold / count if count < threshold else 1.0 for count in class_counts], dtype=torch.double)
        self.total_size = threshold * len(np.unique(np.array(dataset.targets)))

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(torch.multinomial(self.weights, self.total_size, replacement=True).tolist())

    def __len__(self):
        return self.total_size

def get_data_loaders(data_dir, batch_size, train, rank, world_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(),
        transforms.Resize((512, 512)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.RandomErasing(p=0.2, value='random')
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    if train:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None  # Test sampler is not required to be distributed
    
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    return loader

def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, rank, world_size, checkpoint_path):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch}", disable=rank != 0):
            inputs = inputs.cuda(rank, non_blocking=True)
            labels = labels.cuda(rank, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_loader.dataset)
        if rank == 0:
            print(f'Epoch {epoch}/{num_epochs - 1} - Loss: {epoch_loss:.4f}')

            # Save the model if it has the best loss so far
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_path = os.path.join(checkpoint_path, f'best_model_epoch_{epoch}_loss_{best_loss:.4f}.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved best model to {best_model_path}')

def main(rank, world_size):
    setup(rank, world_size)

    train_loader = get_data_loaders(data_path, batch_size, rank, world_size)

    model = models.efficientnet_v2_l(pretrained=False)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    model = model.cuda(rank)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5*len(train_loader), mode='triangular')

    train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, rank, world_size, checkpoint_path)

    if rank == 0:
        final_model_path = os.path.join(checkpoint_path, 'final_model.pth')
        torch.save(model.state_dict(), final_model_path)
        print(f'Final model saved to {final_model_path}')

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)