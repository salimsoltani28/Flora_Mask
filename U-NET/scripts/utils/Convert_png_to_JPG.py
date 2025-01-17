from PIL import Image
import os

# Path to the folder containing PNG images
input_folder = "/mnt/gsdata/projects/bigplantsens/1_Flora_mask/01_MyDiv/Data/5_iNaturalist_myDiv_tree_species_filtered_by_month/grass/Grass_ortho/"
output_folder = "/mnt/gsdata/projects/bigplantsens/1_Flora_mask/01_MyDiv/Data/5_iNaturalist_myDiv_tree_species_filtered_by_month/grass/Grass_ortho_jpg"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through each file in the input folder
file_list = sorted(os.listdir(input_folder))  # Ensure original order
image_counter = 1  # Start numbering from 1

for filename in file_list:
    if filename.endswith(".png"):  # Check if the file is a PNG
        # Open the PNG image
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)

        # Convert to RGB (to ensure compatibility with JPG format)
        img = img.convert("RGB")

        # Save the image as JPG in the output folder with new name
        new_filename = f"Grass_{image_counter}.jpg"  # Rename to Grass_X.jpg
        output_path = os.path.join(output_folder, new_filename)
        img.save(output_path, "JPEG")

        print(f"Converted and Renamed: {filename} -> {new_filename}")
        image_counter += 1

print("Conversion and renaming complete!")
