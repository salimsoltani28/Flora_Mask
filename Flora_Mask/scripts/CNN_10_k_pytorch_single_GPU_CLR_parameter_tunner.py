import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Sampler
import numpy as np
from tqdm import tqdm
import logging
import copy
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter
import optuna

# Paths and constants
checkpoint_path = "/mnt/ssd2/ms2487/Workshop_home/1_Flora_mask/2_myDiv/checkpoints/CNN_updated_code_withValidation/"
data_path = "/home/ms2487/workshop/Flora_Mask/data/Labeled_data_seprated_in_Folder/image/"
num_img_per_class = 4000

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class CustomBalancedSampler(Sampler):
    """A custom sampler that oversamples classes with fewer than threshold samples."""
    def __init__(self, dataset, threshold=num_img_per_class):
        self.indices = list(range(len(dataset)))
        self.num_samples = threshold * len(np.unique(np.array(dataset.targets)))
        _, class_counts = np.unique(np.array(dataset.targets), return_counts=True)
        self.weights = []
        for count in class_counts:
            if count < threshold:
                self.weights += [threshold / count] * count
            else:
                self.weights += [1.0] * count
        self.weights = torch.tensor(self.weights, dtype=torch.double)

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples

def prepare_device():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    return device

def get_data_loaders(data_dir, batch_size, train):
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
        sampler = CustomBalancedSampler(dataset)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=4, shuffle=False)
        return train_loader
    else:
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        return test_loader

def train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs, device, writer, trial):
    # Assuming validation logic is implemented here and returns validation accuracy
    # For simplicity, we're using training accuracy as a placeholder
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_corrects += torch.sum(preds == labels.data).item()
        
        epoch_acc = running_corrects / len(train_loader.dataset)
        writer.add_scalar('Training Accuracy', epoch_acc, epoch)
        
        # Report intermediate objective value.
        trial.report(epoch_acc, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if epoch_acc > best_acc:
            best_acc = epoch_acc
    
    return best_acc

def objective(trial):
    device = prepare_device()
    train_loader = get_data_loaders(data_path, 16, True)  # Fixed batch size for simplicity
    
    model = models.efficientnet_v2_l(pretrained=False)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 10)  # Adjust for your number of classes
    model.to(device)

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    momentum = trial.suggest_uniform('momentum', 0.4, 0.99)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()
    scheduler = CyclicLR(optimizer, base_lr=lr/10, max_lr=lr, step_size_up=2500, mode='triangular')

    writer = SummaryWriter(log_dir=f'runs/CNN_10k_heatmap_trial_{trial.number}')
    best_acc = train_model(model, criterion, optimizer, scheduler, train_loader, 10, device, writer, trial)  # Reduced epochs for example
    writer.close()
    return best_acc

def main():
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
