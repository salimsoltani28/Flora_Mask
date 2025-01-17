# Set environment variables to limit TensorFlow to 10 CPUs
No_cpus="1"
# Set environment variables to limit CPU usage
Sys.setenv("OMP_NUM_THREADS" = No_cpus)
Sys.setenv("MKL_NUM_THREADS" = No_cpus)
Sys.setenv("LOKY_MAX_CPU_COUNT" = No_cpus)

#Rscript your_script.R <checkpoint_name> <start_tile> <end_tile> [<shift_value>]
library(reticulate)
# Setup Environment
reticulate::use_condaenv(condaenv = "tfr_terra")
.libPaths("/home/ms2487/miniconda3/envs/tfr_terra/lib/R/library")



require(raster)
require(keras)
library(tensorflow)
require(rgdal)
# Set the environment variable
Sys.setenv(SM_FRAMEWORK = "tf.keras")

# Load segmentation_models
sm <- import("segmentation_models")

#require(rgeos)

#find and select the GPUs
gpu_indices <- c(1, 3)
gpu_devices <- tf$config$list_physical_devices()[gpu_indices]
tf$config$set_visible_devices(gpu_devices)
# 
gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)
# List all physical devices
all_devices <- tf$config$list_physical_devices('GPU')



# Script Parameters from Command Line Arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  stop("Usage: Rscript script.R <checkpoint_name> <start_tile> <end_tile> [<shift_value>]")
}

checkpoint_name <- args[1]  # First argument: checkpoint name
start_tile <- as.integer(args[2])  # Second argument: start tile
end_tile <- as.integer(args[3])  # Third argument: end tile
shift_value <- if (length(args) >= 4) as.integer(args[4]) else 0  # Optional fourth argument: shift value, default to 0

# Set the fixed prediction threshold
pred_threshold <- 0.3


# Base workdir path
workdir <- "/mnt/gsdata/projects/bigplantsens/2_UNET_on_Flora_Mask/01_MyDiv/"
ortho_image_file <- file.path(workdir, "Data/UAV_Orthoimags/Ortho_July/02_Orthoimage_July_2022_Mosaic.tif")
script_dir <- file.path(workdir, "scripts/utils")
outputfolder <- file.path(workdir, "Pred_output/")

# Use the user-provided checkpoint name to define both checkpoint_dir and pred_dir_name
checkpoint_dir <- file.path(workdir, "checkpoints", checkpoint_name)
pred_folder <- file.path(outputfolder,"Ortho_July_" ,checkpoint_name,"/")

# Check if the output directory exists before creating it
if (!dir.exists(pred_folder)) {
  dir.create(pred_folder, recursive = TRUE)
}

# # Load Customized Loss Function based on Background Class Ignored
# message("Did you ignore the background class during training? (yes/no): ")
# bg_ignored <- readline(prompt = "Enter yes or no: ")

#if (tolower(bg_ignored) == "no") {
  # If the background class was ignored
  source(file.path(script_dir, "Customized_loss_function_Unet_NA_ignored.R"))
  #message("Using standard loss function.")
#} else {
  # If the background class was not ignored
  
  # source(file.path(script_dir, "Customized_loss_function_Unet.R"))
  # message("Using loss function with background class ignored.")

#}
#hdf5
#keras


model_files <- list.files(checkpoint_dir, pattern = "weights.*keras")
if (length(model_files) == 0) {
  stop("No model files found in the specified checkpoint directory: ", checkpoint_dir)
}
loss_values <- sapply(model_files, function(file) {
  as.numeric(unlist(strsplit(unlist(strsplit(file, "-"))[2], ".keras"))[1])
})
best_model_file <- model_files[which.min(loss_values)]

# Load the model
model <- load_model_hdf5(file.path(checkpoint_dir, best_model_file), compile = FALSE,
custom_objects = list(
  "bce_dice_loss" = sm$losses$bce_dice_loss,
  "dice_coef" = dice_coef))
summary(model)

#print best model file
print(best_model_file)
# Print the provided arguments
message("Using checkpoint: ", checkpoint_name)
message("Using prediction threshold: ", pred_threshold)
message("Processing tiles from: ", start_tile, " to ", end_tile)
message("Shift value: ", shift_value)

# Compile the model
model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.0001),
  loss = bce_dice_loss,
  metrics = custom_metric("dice_coef", dice_coef)
)

# Load the orthoimage
if (!file.exists(ortho_image_file)) {
  stop("Raster file does not exist at the specified path: ", ortho_image_file)
}
ortho1 <- raster::stack(ortho_image_file)

# Crop the ortho into small chunks for easy prediction
no_col <- 2
no_row <- 20
res <- 512L  # Resolution of the tiles
chnl <- 3L   # Number of channels (RGB)

res_row <- floor(dim(ortho1)[1] / no_row)
res_col <- floor(dim(ortho1)[2] / no_col)

# Expand the division over the ortho
ind_col_ortho <- cbind(seq(1, floor((dim(ortho1)[2]) / res_col) * res_col, res_col))
ind_row_ortho <- cbind(seq(1, floor((dim(ortho1)[1]) / res_row) * res_row, res_row))
ind_grid_ortho <- expand.grid(ind_col_ortho, ind_row_ortho)
dim(ind_grid_ortho)

# Initialize progress bar
total_tiles <- end_tile - start_tile + 1
pb <- txtProgressBar(min = 0, max = total_tiles, style = 3)

# Process the tiles from start_tile to end_tile
for (ii in start_tile:end_tile) {
  ortho <- raster::crop(ortho1, extent(ortho1,
    ind_grid_ortho[ii, 2], ind_grid_ortho[ii, 2] + res_row + 1000L,
    ind_grid_ortho[ii, 1], ind_grid_ortho[ii, 1] + res_col + 1000L))

  ind_col <- cbind(seq(1, floor(dim(ortho)[2] / res) * res, floor(res))) + shift_value
  ind_row <- cbind(seq(1, floor(dim(ortho)[1] / res) * res, floor(res))) + shift_value
  ind_grid <- expand.grid(ind_col, ind_row)

  # Set an empty raster for predictions based on ortho
  predictions <- setValues(ortho[[1]], NA)

  # Moving window prediction
  for (i in 1:nrow(ind_grid)) {
    ortho_crop <- raster::crop(ortho, extent(ortho, ind_grid[i, 2], ind_grid[i, 2] + res - 1, ind_grid[i, 1], ind_grid[i, 1] + res - 1))
    ortho_crop <- tf$convert_to_tensor(as.array(ortho_crop) / 255) %>%
      tf$keras$preprocessing$image$smart_resize(size = c(res, res)) %>%
      tf$reshape(shape = c(1L, res, res, chnl))

    if (length(which(is.na(ortho_crop) == TRUE)) == 0) {
      # Conservative prediction
      predicted_values <- predict(model, ortho_crop)[1, , , ]
      predicted_values <- ifelse(predicted_values < pred_threshold, NA, predicted_values)
      predictions[ind_grid[i, 2]:(ind_grid[i, 2] + res - 1), ind_grid[i, 1]:(ind_grid[i, 1] + res - 1)] <- t(as.array(k_argmax(predicted_values)))
    }
  }

  # Save the predictions
  writeRaster(predictions, filename = paste0(pred_folder, ii, "_Unet_pred_wholeortho","_",shift_value ,".tif"), overwrite = TRUE)

  # Update progress bar
  setTxtProgressBar(pb, ii - start_tile + 1)
}

# Close the progress bar after completion
close(pb)

message("Processing completed for tiles: ", start_tile, " to ", end_tile)
