

original_images <- list.files("/net/scratch/ssoltani/knossos_data/workshop/11_FloraMask/1_mask_generation/dataset/04_PlantNet_data_old_delete/011_Grass_fromGoogle_planet",pattern = ".jpg", recursive = T, full.names = T)


# Install packages if not already installed
# install.packages(c("raster", "rgdal", "magick"))

# Load required libraries
library(raster)
library(magick)
library(rgdal)
library(magick)

# Set the directory containing your images
image_dir <- "/net/scratch/ssoltani/knossos_data/workshop/11_FloraMask/1_mask_generation/dataset/04_PlantNet_data_old_delete/grass_images"

# Set the output directory
output_dir <- "/net/scratch/ssoltani/knossos_data/workshop/11_FloraMask/1_mask_generation/dataset/04_PlantNet_data_old_delete/grass_masks"

# List all PNG files in the directory (change pattern as needed for other file types)
image_files <- original_images

# Loop through each file
for(i in seq_along(image_files)) {
  # Load the image using magick
  img <- image_read(image_files[i])
  
  # Get image dimensions
  img_info <- image_info(img)
  nrow <- img_info$height
  ncol <- img_info$width
  
  # Initialize a raster with the same dimensions
  r <- raster(nrows = nrow, ncols = ncol, xmn=0, xmx=ncol, ymn=0, ymx=nrow)
  
  # Set all values to 11
  r[] <- 10
  #convert the pred to png
  ortho_pred <- as.array(r)
  ortho_pred <- image_read(ortho_pred/255)
  
  # Define the output file names, using the same base name but different directories/extension as needed
  raster_output_path <- file.path(output_dir, paste0("mask_Grass_",i, ".png"))
  image_output_path <- file.path(image_dir, paste0("Grass_",i, ".jpg"))
  
  # Save the raster (mask) as PNG
  image_write(ortho_pred,format = "png",path =  raster_output_path)
  
  # Save the image in the output directory (re-saving the original image)
  image_write(img, path = image_output_path, format = "jpg")
  
 
}
