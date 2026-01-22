import os
import torch
import random
import cv2
import numpy as np
from PIL import Image
from torchvision import models
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.image import preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import warnings
warnings.filterwarnings('ignore')

def get_random_points_within_contour(contour, num_points):
    rect = cv2.boundingRect(contour)
    points = []
    while len(points) < num_points:
        x = random.randint(rect[0], rect[0] + rect[2])
        y = random.randint(rect[1], rect[1] + rect[3])
        if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
            points.append((x, y))
    return points

# Load your custom model
model = models.efficientnet_v2_l(weights=None)
num_classes = 11
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# Load model weights
model.load_state_dict(torch.load('/yourpath//Flora_Mask/2_myDiv/checkpoints/checkpoint/best_model_110_0.31.pth'))

# Set device to GPU:1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move model to GPU:1
model = model.to(device)
model.eval()

# Define the target layer for Grad-CAM
target_layers = [model.features[-1]]

# Path to the folder with images
folder_path = '/yourpath//Flora_Mask/2_myDiv/data/data_check/First_class/'
save_folder = '/yourpath//Flora_Mask/2_myDiv/data/data_check/output/'

# Ensure the save folder exists
os.makedirs(save_folder, exist_ok=True)

# Process each image in the folder
for image_name in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_name)
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size
    original_image_np = np.array(original_image) / 255.0
    preprocessed_image = preprocess_image(original_image)

    # Move tensor to GPU:1
    input_tensor = preprocessed_image.to(device)

    # Create a Grad-CAM object and generate the CAM mask
    cam = XGradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_class = 0  # Adjust this as needed
    targets = [ClassifierOutputTarget(target_class)]
    
    with torch.no_grad():
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    
    grayscale_cam = grayscale_cam[0, :]

    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(grayscale_cam, original_size, interpolation=cv2.INTER_LINEAR)
    heatmap_resized = np.uint8(255 * heatmap_resized)

    # Apply colormap and invert the heatmap
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_color = cv2.bitwise_not(heatmap_color)  # Invert the heatmap colors

    # Threshold and find contours
    threshold_value = 150  # Adjust as needed
    _, thresholded = cv2.threshold(heatmap_resized, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Combine the inverted heatmap with the original image
    original_image_scaled = np.float32(original_image_np)
    heatmap_scaled = np.float32(heatmap_color) / 255
    visualization = cv2.addWeighted(original_image_scaled, 0.6, heatmap_scaled, 0.4, 0)

    # Draw contours and points on the combined image
    for contour in contours:
        # Draw contour
        cv2.drawContours(visualization, [contour], -1, (255, 0, 0), 2)  # Blue contour

        # Get 10 random points within the contour
        points = get_random_points_within_contour(contour, 10)

        # Draw points
        for point in points:
            cv2.circle(visualization, point, 5, (0, 255, 0), -1)  # Green points

    # Convert the visualization to a suitable format for saving
    visualization = np.uint8(255 * visualization)

    # Save visualization
    vis_image = Image.fromarray(visualization)
    vis_save_path = os.path.join(save_folder, f'vis_{image_name}')
    vis_image.save(vis_save_path)
    print(f"Visualization saved to {vis_save_path}")

    # Free up memory after processing each image
    del input_tensor, grayscale_cam
    torch.cuda.empty_cache()
