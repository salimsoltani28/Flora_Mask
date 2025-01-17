import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from tqdm import tqdm

# ================================
# Hyperparameters and Config
# ================================
NUM_CLASSES = 11
IN_CHANNELS = 3
NUM_EPOCHS = 150
BATCH_SIZE = 15
LEARNING_RATE = 1e-4
TRAIN_VAL_SPLIT = 0.12
RANDOM_SEED = 42
IMAGE_SIZE = 512
SAMPLE_SIZES_REPLACE = 4000
MODEL_SAVE_DIR = "/net/scratch/ssoltani/knossos_data/workshop/11_FloraMask/2_Unet_on_flora_mask/2_MyDiv/checkpoints/New_Data_unet_pytorch_check/"
DATA_PATH_METADATA = "/net/scratch/ssoltani/knossos_data/workshop/11_FloraMask/2_Unet_on_flora_mask/2_MyDiv/Data/MyDiv_plantNet_Distance_Stem_filter.csv"
GRASS_DIR = "/net/scratch/ssoltani/knossos_data/workshop/11_FloraMask/1_Flora_mask/2_MyDiv/data/"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ================================
# Data Loading and Preprocessing
# ================================
# Load data
data_path_metadata = pd.read_csv(DATA_PATH_METADATA)

# Filter out images
all_images_filtered = data_path_metadata[(data_path_metadata['stam_nostam'] == 0) & 
                                         (data_path_metadata['dist'] < 15) & 
                                         (data_path_metadata['pixel_percent'] > 40)][['img', 'mask', 'ref']]

# Load grass data
grass_image = list(Path(GRASS_DIR + "grass_images").glob("*.jpg"))
grass_mask = list(Path(GRASS_DIR + "grass_masks").glob("*.png"))
grass_label = [11] * len(grass_mask)
grass_data = pd.DataFrame({'img': grass_image, 'mask': grass_mask, 'ref': grass_label})

# Combine with reference data
all_images_filtered = pd.concat([all_images_filtered, grass_data])

# Balance the data for all classes
def sample_data(group):
    if len(group) < SAMPLE_SIZES_REPLACE:
        return group.sample(SAMPLE_SIZES_REPLACE, replace=True)
    else:
        return group.sample(SAMPLE_SIZES_REPLACE, replace=False)

balanced_data = all_images_filtered.groupby('ref').apply(sample_data).reset_index(drop=True)

# Split data into train and validation
train_data, val_data = train_test_split(balanced_data, test_size=TRAIN_VAL_SPLIT, random_state=RANDOM_SEED)

# ================================
# Custom Dataset
# ================================
class CustomDataset(Dataset):
    def __init__(self, dataframe, image_size=512, no_classes=0, train=True):
        self.dataframe = dataframe
        self.image_size = image_size
        self.no_classes = no_classes
        self.train = train
        self.basic_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.target_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['img']
        mask_path = self.dataframe.iloc[idx]['mask']
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.train:
            image = TF.adjust_brightness(image, brightness_factor=0.1)
            image = TF.adjust_contrast(image, contrast_factor=np.random.uniform(0.9, 1.1))
            image = TF.adjust_saturation(image, saturation_factor=np.random.uniform(0.9, 1.1))
            image = TF.adjust_hue(image, hue_factor=np.random.uniform(-0.02, 0.02))
        
        image = self.basic_transforms(image)
        mask = self.target_transform(mask)
        
        mask = np.array(mask)
        mask = mask - 1
        mask = torch.from_numpy(mask)
        mask = np.clip(mask, 0, self.no_classes - 1)
        mask = torch.eye(self.no_classes)[mask.long()]
        
        return image, mask.permute(2, 0, 1)

def create_dataloader(dataframe, image_size, batch_size, epochs, shuffle=True, no_classes=0, train=True):
    dataset = CustomDataset(dataframe, image_size=image_size, no_classes=no_classes, train=train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

train_dataloader = create_dataloader(dataframe=train_data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, shuffle=True, no_classes=NUM_CLASSES, train=True)
val_dataloader = create_dataloader(dataframe=val_data, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, shuffle=False, no_classes=NUM_CLASSES, train=False)

# ================================
# Dice Loss and Coefficient Functions
# ================================
def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

# ================================
# Model, Optimizer, and Scheduler
# ================================
model = smp.Unet(encoder_name="efficientnet-b7", classes=NUM_CLASSES, activation="softmax").to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LEARNING_RATE, max_lr=1e-3, step_size_up=2000, mode='triangular')

# ================================
# Training Loop
# ================================
best_val_dice = float('-inf')

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = []
    train_dice_scores = []

    train_progress = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Training]')
    for images, masks in train_progress:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        outputs = torch.softmax(outputs, dim=1)
        loss = dice_loss(outputs, masks, multiclass=True)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss.append(loss.item())
        dice_score = multiclass_dice_coeff(outputs, masks)
        train_dice_scores.append(dice_score.item())

        train_progress.set_postfix(loss=np.mean(train_loss), dice_score=np.mean(train_dice_scores))

    avg_train_loss = np.mean(train_loss)
    avg_train_dice = np.mean(train_dice_scores)

    model.eval()
    val_loss = []
    val_dice_scores = []

    val_progress = tqdm(val_dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Validation]')
    with torch.no_grad():
        for images, masks in val_progress:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)
            loss = dice_loss(outputs, masks, multiclass=True)

            val_loss.append(loss.item())
            dice_score = multiclass_dice_coeff(outputs, masks)
            val_dice_scores.append(dice_score.item())

            val_progress.set_postfix(val_loss=np.mean(val_loss), val_dice_score=np.mean(val_dice_scores))

    avg_val_loss = np.mean(val_loss)
    avg_val_dice = np.mean(val_dice_scores)

    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.2f}, Train Dice: {avg_train_dice:.2f}, Val Loss: {avg_val_loss:.2f}, Val Dice: {avg_val_dice:.2f}')

    # Save checkpoint
    checkpoint_filename = f'model_epoch_{epoch+1}_val_loss_{avg_val_loss:.2f}_val_dice_{avg_val_dice:.2f}.pth'
    checkpoint_path = os.path.join(MODEL_SAVE_DIR, checkpoint_filename)
    torch.save(model.state_dict(), checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_filename}')

    # Save the best model
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        best_model_path = os.path.join(MODEL_SAVE_DIR, f'best_model_epoch_{epoch+1}_val_loss_{avg_val_loss:.2f}_val_dice_{avg_val_dice:.2f}.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f'Best model updated: {best_model_path}')
