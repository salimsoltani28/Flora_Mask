import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from segment_anything import sam_model_registry, SamPredictor
import random
import torch.nn as nn
from collections import OrderedDict
# Function to sample fixed number of points within a contour
def sample_points_within_contour(contour, num_points):
    rect = cv2.boundingRect(contour)
    mask = np.zeros((rect[3], rect[2]), dtype=np.uint8)
    shifted_contour = contour - np.array([[rect[0], rect[1]]])
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(mask == 255)
    if len(xs) < num_points:
        return [(xs[i] + rect[0], ys[i] + rect[1]) for i in range(len(xs))]
    sampled_indices = random.sample(range(len(xs)), num_points)
    return [(xs[i] + rect[0], ys[i] + rect[1]) for i in sampled_indices]

# Setup base directory and parameters
base_dir = '/mnt/FS_data/ms2487/workshop/1_Flora_mask/00_data/2_Mt_Baldy_iNat_Data/'  # Update this to your base directory path
Threshold_value = 150
No_of_sampled_points = 2  # Number of points to sample within each contour

# Load models and preprocessing
model_path = '/mnt/FS_data/ms2487/workshop/1_Flora_mask/1_checkpoints/best_model_56_0.36.pth'
sam_checkpoint = '/home/ms2487/workshop/1_flora_mask_old/scripts/sam_vit_h_4b8939.pth'
#model = models.efficientnet_v2_l(weights=None)

# Load the model architecture
model = models.efficientnet_v2_l(pretrained=False)  # No pretrained weights

# Modify the classifier to match the number of classes in your dataset
num_ftrs = model.classifier[1].in_features  # Adjust this index if necessary
model.classifier[1] = nn.Linear(num_ftrs, 10)  # Assuming 11 is the correct number of classes

# Now load the checkpoint
checkpoint = torch.load(model_path, map_location='cpu')  # Adjust map_location as necessary

# Adjust for the DDP 'module.' prefix if present, as described previously
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    name = k[7:] if k.startswith('module.') else k  # Remove 'module.' prefix
    new_state_dict[name] = v

# Load the adjusted state dict into the model
model.load_state_dict(new_state_dict)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Move the model to the appropriate device and set it to evaluation mode
model = model.to(device)
model.eval()
#model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 11)
#model.load_state_dict(torch.load(model_path))
#model.eval()
#model.to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))

sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define patterns for the file types to include
patterns = tuple(['.jpg', '.png', '.JPEG', '.JPG', '.PNG', '.jpeg'])
# Iterate over each sub-directory in the base directory and update target_class for each
for folder_idx, subdir in enumerate(sorted(os.listdir(base_dir))):
    target_class = folder_idx  # Update target_class for each folder
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        save_folder = f'{subdir_path}_mask'
        os.makedirs(save_folder, exist_ok=True)

        
        for image_name in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, image_name)
            # Check if the file is an image and has thse appropriate extension
            if not os.path.isfile(image_path) or not image_name.lower().endswith(patterns):
                continue  # Skip the rest of the loop body for this iteration
            original_image = Image.open(image_path).convert('RGB')
            # Assuming 'transform' is defined elsewhere and prepares the image for your needs
            input_tensor = transform(original_image).unsqueeze(0)
            # Now you can do something with 'input_tensor', like listing, processing, etc.

            # Generate CAM and apply threshold to find contours
            cam = GradCAM(model=model, target_layers=[model.features[-1]])
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0, :]
            grayscale_cam_resized = cv2.resize(grayscale_cam, original_image.size, interpolation=cv2.INTER_LINEAR)
            _, binary_map = cv2.threshold(np.uint8(255 * grayscale_cam_resized), Threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            all_sampled_points, all_input_labels = [], []
            for contour in contours:
                sampled_points = sample_points_within_contour(contour, No_of_sampled_points)
                all_sampled_points.extend(sampled_points)
                all_input_labels.extend([1] * len(sampled_points))

            if all_sampled_points:
                # Initialize a list to hold all generated masks
                all_masks = []

                # Loop to sample points and generate masks 10 times
            
                # Sample points again for each iteration to generate different masks
                # Sample points again for each iteration to generate different masks
                temp_sampled_points = []
                for contour in contours:
                    temp_sampled_points.extend(sample_points_within_contour(contour, No_of_sampled_points))

                # Set the image for prediction
                predictor.set_image(np.array(original_image))

                # First prediction to generate multiple masks and select the best one based on scores
                masks, scores, logits = predictor.predict(
                point_coords=np.array(temp_sampled_points),
                point_labels=np.array(all_input_labels, dtype=np.int32),
                multimask_output=True
                )

                # Select the best mask based on the highest score
                best_mask_index = np.argmax(scores)
                best_mask_input = logits[best_mask_index, :, :]

                # Use the best mask for final prediction
                final_mask, _, _ = predictor.predict(
                point_coords=np.array(temp_sampled_points),
                point_labels=np.array(all_input_labels, dtype=np.int32),
                mask_input=best_mask_input[None, :, :],
                multimask_output=False
                )

                # Ensure final_mask is a 2D array for image saving
                final_mask = np.squeeze(final_mask)
                
                # Change True to folder_idx and False to 10
                # Note: Assuming final_mask is a boolean mask; if not, adjust the condition accordingly.
                modified_mask = np.where(final_mask, folder_idx, 10)
                
                mask_save_path = os.path.join(save_folder, f'mask_{os.path.splitext(image_name)[0]}.png')
                #mask_save_path = os.path.join(save_folder, f'mask_{image_name}')
                # Save the modified mask where True is replaced by folder_idx and False by 10
                modified_mask_uint8 = modified_mask.astype(np.uint8)
                cv2.imwrite(mask_save_path, modified_mask_uint8)
                print(f"Combined mask modified and saved to {mask_save_path}")
            else:
                print(f"No contours found for {image_name}, skipping mask generation.")




