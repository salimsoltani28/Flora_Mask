
# Set environment variables to limit TensorFlow to 10 CPUs
No_cpus="8"
# Set environment variables to limit CPU usage
Sys.setenv("OMP_NUM_THREADS" = No_cpus)
Sys.setenv("MKL_NUM_THREADS" = No_cpus)
Sys.setenv("LOKY_MAX_CPU_COUNT" = No_cpus)

# Set Library Paths
library(reticulate)
# Setup Environment
reticulate::use_condaenv(condaenv = "tfr215",required = T)
#.libPaths("/home/ms2487/miniconda3/envs/tfr215/lib/R/library")

# Load Libraries
libraries_to_load <- c(
  "reticulate", 
  "keras", 
  "tensorflow", 
  "tidyverse", 
  "tibble", 
  "rsample", 
  #"magick", 
  "ggplot2", 
  "gtools"
)
library(tensorflow)
lapply(libraries_to_load, library, character.only = TRUE)
# set a memory limit
memory.limit(size = 115000)
#set seeds
set.seed(512)
# Set TensorFlow seed
tf$random$set_seed(as.integer(42))

# WandB Initialization
wandb <- import("wandb")
wandb$login(key = "753058fac9941b142125245452790d1caf6fa227")

# List all available GPU devices
# set memory growth policy
# List all available GPU devices
all_devices <- tf$config$list_physical_devices('GPU')

# Define the GPU index you want to use as a variable
gpu_index <- 0  # Change this to the desired GPU index (e.g., 1, 2, etc.)

# Get all available GPU devices
all_devices <- tf$config$list_physical_devices('GPU')

# Check if at least one GPU is available
if (length(all_devices) > 0) {
  # Ensure the specified GPU index is within the range of available GPUs
  if (gpu_index < length(all_devices)) {
    # Restrict TensorFlow to only use the specified GPU
    selected_gpu <- all_devices[[gpu_index + 1]]  # R is 1-based, so add 1
    
    tf$config$set_visible_devices(selected_gpu, 'GPU')
    
    # Enable memory growth on the specified GPU
    tf$config$experimental$set_memory_growth(selected_gpu, TRUE)
    
    message("Using GPU: ", selected_gpu$name)
  } else {
    message("Error: Specified GPU index (", gpu_index, ") is out of range. Only ", length(all_devices), " GPUs available.")
  }
} else {
  message("No GPU devices found.")
}

# Uncomment if you want to enable eager execution
# tfe_enable_eager_execution(device_policy = "silent")

# Uncomment if you are using MirroredStrategy
# strategy <- tf$distribute$MirroredStrategy()
# strategy$num_replicas_in_sync

# Set Parameters
tilesize <- 512L
chnl <- 3L
no_epochs <- 80L
no_classes <- 11L # one more class for baren land
batch_size <- 10L
#garss_class_label=12L #the SAM background class number is 10
class_ignored = 15L #out of distribution
sample_sizes_replace <- 4000
# Set Working Directory
workdir <- "/mnt/gsdata/projects/bigplantsens/2_UNET_on_Flora_Mask/01_MyDiv/"
setwd(workdir)


# Load Utility Functions
source("scripts/utils/Create_dataset_function.R")
source("scripts/utils/Customized_loss_function_Unet_Background_ignored.R")
source("scripts/utils/Unet_model_Gelu.R")


# 

data_path_metadata <- read.csv("/mnt/ssds/ms2487/Mydiv_Unet_training_data/5_iNaturalist_myDiv_tree_species_filtered_by_month_60pZoomed_Ortho_Background/MyDiv_monthlydata_plantNet_Distance_Stem_filter_modified_60P_Zoomed_OrthoBack_moved.csv") 



# # Plotting the histogram of pixel_percent for each class (ref) using facets
# ggplot(all_imags_filtered, aes(x = pixel_percent)) +
#   geom_histogram(binwidth = 5, color = "black", fill = "steelblue") +
#   labs(title = "Histogram of Pixel Percent by Class",
#        x = "Pixel Percent",
#        y = "Count") +
#   theme_minimal() +
#   facet_wrap(~ ref, scales = "free_y") # Create a facet for each class (ref)

# #filter out images with distance
all_imags_filtered <- data_path_metadata %>%
  #set all dist for grass to 5
  #filter(!ref==11) %>%
     filter(stam_nostam==0 ,dist>0.2,dist<20, pixel_percent>30) %>% dplyr::select(img, mask,ref) #%>%
    #filter(stam_nostam==0, dist>0.2 ,dist<20, pixel_percent>30, !grepl("modified", mask)) %>% dplyr::select(img, mask,ref) #%>%
#filter(stam_nostam==0, pixel_percent>50) %>% select(img, mask,ref)
#filter(stam_nostam==0  ,dist<15) %>% select(img, mask,ref) #%>%
table(data_path_metadata$ref)/2
table(all_imags_filtered$ref)/2
length(data_path_metadata$ref)/2
length(all_imags_filtered$ref)/2
#model name
model_dir_name <- "EXP3.3_over0.2under20_stem_30percent_cls11_Ortho_backRplcd_ZMEXCLD_Adamw_gelu_img512_"

############################################sample the number of images to combine with PlanNet data
#load grass data
# grass_dir="Data/5_iNaturalist_myDiv_tree_species_filtered_by_month/grass/"
# grass_image= mixedsort(list.files(paste0(grass_dir, "grass_images"), pattern = "jpg", full.names = T))
# grass_mask= mixedsort(list.files(paste0(grass_dir, "grass_masks"), pattern = "png", full.names = T))
#grassdata = tibble(img=grass_image, mask=grass_mask, ref=garss_class_label)

####combine it with the refrence data
# all_imags_filtered <- rbind(all_imags_filtered,grassdata)
# table(all_imags_filtered$ref)


table(all_imags_filtered$ref)

#Balance the data for all classes





#sample with replacment
iNatData_withreplac <- all_imags_filtered %>%
  group_by(ref) %>%
  filter(!n()>sample_sizes_replace) %>%
  nest() %>%
  ungroup() %>%
  mutate(n=sample_sizes_replace) %>%
  mutate(samp=map2(data, n,sample_n,replace=T)) %>%
  select(-c(data,n)) %>%
  unnest(cols = c(samp))

iNatData_no_replace<- all_imags_filtered %>%
  group_by(ref) %>%
  filter(!n()<sample_sizes_replace) %>%
  nest() %>%
  ungroup() %>%
  mutate(n=sample_sizes_replace) %>%
  mutate(samp=map2(data, n,sample_n)) %>%
  select(-c(data,n)) %>%
  unnest(cols = c(samp))
#check the data
table(iNatData_withreplac$ref)
table(iNatData_no_replace$ref)

combined_data <- rbind(iNatData_withreplac,iNatData_no_replace)
table(combined_data$ref)
#check for duplicates
# Then, count the number of duplicates for each class
# Step 1: Identify duplicates
testdata <- combined_data %>%
  group_by(ref) %>%
  # Flag duplicates as before
  mutate(is_duplicate = duplicated(img) | duplicated(img, fromLast = TRUE)) %>%
  # Count duplicates per class
  summarise(num_duplicates = sum(is_duplicate, na.rm = TRUE)) %>%
  # Replace NA with 0 if necessary (though sum(is_duplicate) should not produce NA)
  replace_na(list(num_duplicates = 0)) %>%
  ungroup()

print(testdata)








#check the length of images and mask
path_img <- combined_data$img
path_msk <- combined_data$mask
#remove combined data
remove(data_path_metadata,all_imags_filtered,combined_data,iNatData_no_replace,iNatData_withreplac)

length(path_img)
length(path_msk)


# Check for duplicates again
table(duplicated(path_img))
table(duplicated(path_msk))



# Loading Data ----------------------------------------------------------------


valIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/8), replace = F)
val_img = path_img[valIdx]; val_msk = path_msk[valIdx]
train_img = path_img[-valIdx]; train_msk = path_msk[-valIdx]

train_data = tibble(img = train_img,
                    msk = train_msk)
val_data = tibble(img = val_img,
                  msk = val_msk)
dataset_size <- length(train_data$img)




# Parameters ----------------------------------------------------------------


dataset_size <- length(train_data$img)


training_dataset <- create_dataset(train_data, train = TRUE, batch = batch_size, epochs = no_epochs, dataset_size = dataset_size,image_size = tilesize,no_classes=no_classes)
validation_dataset <- create_dataset(val_data, train = FALSE, batch = batch_size, epochs = no_epochs,image_size = tilesize,no_classes=no_classes)

dataset_iter = reticulate::as_iterator(training_dataset)
example = dataset_iter %>% reticulate::iter_next()
example[[1]]
example[[2]]
par(mfrow=c(1,2))
plot(as.raster(as.array(example[[1]][1,,,1:3]), max = 1))
#plot(as.raster(as.array(example[[2]][1,,,1]), max = 1))
#################################
# Aggregate the one-hot encoded mask into a single channel mask
# Aggregate the one-hot encoded mask into a single channel mask
single_channel_mask <- tf$argmax(example[[2]][1,,,], axis = as.integer(-1))

# Convert the mask to a numpy array
single_channel_mask_np <- as.array(single_channel_mask)

# Plot the single channel mask
plot(as.raster(single_channel_mask_np, max = max(single_channel_mask_np)))


#with(strategy$scope(), {
model <- get_unet_128()
#}))

#model <- get_unet_128()



#output model name
output_model_name <- paste0(str_replace(model_dir_name,"/",""),no_epochs)
######################Monitoring

run <- wandb$init(
# Set the project where this run will be logged
#project = "citizen_science",
project = output_model_name,
# Track hyperparameters and run metadata
config = list(
learning_rate = lr,
epochs = no_epochs,
tilesize = tilesize,
no_classes = no_classes,
batch_size = batch_size,
sample_sizes_replace =sample_sizes_replace
   )
 )

# # Model fitting ----------------------------------------------------------------
# # 
# checkpoint_dir <- paste0(workdir, "checkpoints/",output_model_name)
# unlink(checkpoint_dir, recursive = TRUE)
# dir.create(checkpoint_dir, recursive = TRUE)
# #hdf5
# filepath = file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.keras")

# cp_callback <- callback_model_checkpoint(filepath = filepath,
#                                          monitor = "dice_coef",
#                                          save_weights_only = FALSE,
#                                          save_best_only = TRUE,
#                                          verbose = 1,
#                                          #mode = "auto",
#                                          mode = "max",
#                                          save_freq = "epoch")

# history <- model %>% fit(x = training_dataset,
#                          epochs = no_epochs,
#                          steps_per_epoch = dataset_size/(batch_size),
#                          callbacks = list(cp_callback, callback_terminate_on_naan(),wandb$keras$WandbCallback()),
#                          validation_data = validation_dataset)
# #





###########################################Run this incase you want to resume the model training


# Model fitting ----------------------------------------------------------------

checkpoint_dir <- paste0(workdir, "checkpoints/",output_model_name)
#unlink(checkpoint_dir, recursive = TRUE)
dir.create(checkpoint_dir, recursive = TRUE)
filepath = file.path(checkpoint_dir, "weights.{epoch:02d}-{val_loss:.2f}.keras")

cp_callback <- callback_model_checkpoint(filepath = filepath,
                                         monitor = "dice_coef",
                                         save_weights_only = FALSE,
                                         save_best_only = TRUE,
                                         verbose = 1,
                                         #mode = "auto",
                                         mode = "max",
                                         save_freq = "epoch")


###resume training
# Load the latest checkpoint
#function to load the last model
load_the_last_model <- function(path) {
  # List all files in the directory
  files <- list.files(path)
  
  # Extract epoch numbers from the filenames
  epoch_numbers <- as.numeric(sub("weights\\.(\\d+)-.*\\.keras", "\\1", files))
  
  # Find the file with the highest epoch number
  max_epoch_index <- which.max(epoch_numbers)
  last_file <- files[max_epoch_index]
  
  # Print the loaded model details
  print(paste0("Loaded model of epoch ", last_file, "."))
  
  # Load the model
  model <- load_model_hdf5(paste0(path, "/", last_file), compile = FALSE)
  
  return(model)
}

# Use the function to load the latest model
latest <- load_the_last_model(checkpoint_dir)

# Define the number of completed epochs
completed_epoch <- 32L
# Load the previously saved weights
weights <- get_weights(latest)
model %>% set_weights(weights)

history <- model %>% fit(x = training_dataset,
                         epochs = no_epochs,
                         initial_epoch = completed_epoch,
                         steps_per_epoch = dataset_size/(batch_size),
                         callbacks = list(cp_callback, callback_terminate_on_naan(),wandb$keras$WandbCallback()),
                         validation_data = validation_dataset)




# # Create a new artifact: save the code
# code_artifact <- wandb$Artifact(
#   name = "training_script",
#   type = "code"
# )

# # Add your script file to the artifact
# code_artifact$add_file("/mnt/gsdata/projects/bigplantsens/1_Flora_mask/01_MyDiv/scripts/01_Training_Unet_Flora_mask.R")  # Replace with the actual path

# # Log the artifact to W&B
# run$log_artifact(code_artifact)

#
# ###finish tracking
wandb$finish()
#
# ###finish tracking
# wandb$finish()
