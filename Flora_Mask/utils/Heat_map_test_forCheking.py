import os
import torch
import random
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from segment_anything import sam_model_registry, SamPredictor
import torch.nn as nn
from collections import OrderedDict

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

num_classes = 10  # Update this to the number of classes in your dataset
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
target_class = 0
# Function to sample fixed number of points within a contour
def sample_points_within_contour(contour, num_points):
    rect = cv2.boundingRect(contour)
    mask = np.zeros((rect[3], rect[2]), dtype=np.uint8)
    shifted_contour = contour - np.array([[rect[0], rect[1]]])
    cv2.drawContours(mask, [shifted_contour], -1, (255), thickness=cv2.FILLED)
    ys, xs = np.where(mask == 255)
    sampled_indices = random.sample(range(len(xs)), min(len(xs), num_points))
    points = [(xs[i] + rect[0], ys[i] + rect[1]) for i in sampled_indices]
    return points


# Load the custom model
model_path = '/home/ms2487/workshop/Flora_Mask/checkpoint/best_model_110_0.31.pth'
checkpoint = torch.load(model_path, map_location=device)
model = models.efficientnet_v2_l(pretrained=False)  # Adjust based on your actual model
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, num_classes)

# Load the state dict directly without adjusting for 'module.' prefix
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()

# SAM setup
sam_checkpoint = "/home/ms2487/workshop/Flora_Mask/checkpoint/model_weights.pth"  # Update this path
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Directories
folder_path = '/mnt/gsdata/users/soltani/Workshop_home_fromSSD2/Workshop_home/Flora_Mask/1_Mt_baldy/data/1_Mt_Baldy_data_check'  # Update this path
save_folder = '/mnt/gsdata/users/soltani/Workshop_home_fromSSD2/Workshop_home/Flora_Mask/2_myDiv/Test_delete/'  # Update this path
os.makedirs(save_folder, exist_ok=True)

# Preprocess function
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

patterns = ('.jpg', '.png', '.jpeg')

for image_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_name)
    if os.path.isfile(image_path) and image_name.lower().endswith(patterns):
        original_image = Image.open(image_path).convert('RGB')
        input_tensor = transform(original_image).unsqueeze(0).to(device)

        # Grad-CAM
        target_layers = [model.features[-1]]
        cam = XGradCAM(model=model, target_layers=target_layers)
        target_class = 7  # Adjust based on your needs
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])[0, :]
        grayscale_cam = cv2.resize(grayscale_cam, original_image.size, interpolation=cv2.INTER_LINEAR)

        # Overlay heatmap on the original image with transparency
        rgb_img = np.array(original_image) / 255.0
        heatmap_overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Convert overlay to BGR for OpenCV operations
        heatmap_overlay_bgr = cv2.cvtColor((heatmap_overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Apply threshold and find contours
        _, binary_map = cv2.threshold(np.uint8(255 * grayscale_cam), 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours and sample points
        for contour in contours:
            cv2.drawContours(heatmap_overlay_bgr, [contour], -1, (0, 255, 0), 2)  # Contour lines
            sampled_points = sample_points_within_contour(contour, 20)  # Sample points
            for point in sampled_points:
                cv2.circle(heatmap_overlay_bgr, point, 5, (0, 0, 255), -1)  # Sampled points

        # Convert back to RGB and save
        final_image_rgb = cv2.cvtColor(heatmap_overlay_bgr, cv2.COLOR_BGR2RGB)
        final_image = Image.fromarray(final_image_rgb)
        combined_save_path = os.path.join(save_folder, f'combined_{image_name}')
        final_image.save(combined_save_path)
        print(f"Combined image saved to {combined_save_path}")