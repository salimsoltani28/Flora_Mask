import os
from PIL import Image
import numpy as np

def create_mask_for_images(source_folder, target_folder):
    # Ensure target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Loop through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            # Open the image file
            img_path = os.path.join(source_folder, filename)
            img = Image.open(img_path)
            
            # Create a mask with the same dimensions and filled with the integer value 3
            maskvalue = 3
            mask = np.full((img.height, img.width), maskvalue, dtype=np.uint8)
            
            # Convert the numpy array to an image
            mask_img = Image.fromarray(mask)
            
            # Save the mask image with the prefix "mask_" and save as PNG
            mask_filename = f"mask_{os.path.splitext(filename)[0]}.png"
            mask_img.save(os.path.join(target_folder, mask_filename))

# Usage
source_folder = '/yourpath//Flora_Mask/2_myDiv/data/image/011_Grass'
target_folder = '/yourpath//Flora_Mask/2_myDiv/data/masks/011_Grass_Mask'
create_mask_for_images(source_folder, target_folder)
