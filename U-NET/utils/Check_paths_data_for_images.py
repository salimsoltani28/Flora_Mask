import pandas as pd
import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Function to check if an image file is valid
def is_valid_image(file_path):
    if not os.path.exists(file_path):
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify the image file integrity
        return True
    except (IOError, SyntaxError):
        return False

# Load the CSV file
csv_file_path = '/mnt/gsdata/projects/bigplantsens/1_Flora_mask/01_MyDiv/Data/5_iNaturalist_myDiv_tree_species_filtered_by_month/MyDiv_monthlydata_plantNet_Distance_Stem_filter_modified_updated.csv'  # Replace with your actual file path
data = pd.read_csv(csv_file_path)

# Extract image and mask paths
img_paths = data['img'].tolist()
mask_paths = data['mask'].tolist()

# Function to process file checks in parallel
def process_file_checks(paths, num_cpus):
    with ThreadPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(is_valid_image, paths))
    return results

# Define the number of CPUs to use
num_cpus = 40  # Adjust to the number of desired CPUs

# Check file validity for both img and mask paths
img_validity = process_file_checks(img_paths, num_cpus)
mask_validity = process_file_checks(mask_paths, num_cpus)

# Count results
img_valid = sum(img_validity)
img_invalid = len(img_validity) - img_valid

mask_valid = sum(mask_validity)
mask_invalid = len(mask_validity) - mask_valid

# Print the summary
print(f"Number of CPUs used: {num_cpus}")
print(f"Valid images: {img_valid}")
print(f"Corrupt or missing images: {img_invalid}")
print(f"Valid masks: {mask_valid}")
print(f"Corrupt or missing masks: {mask_invalid}")
