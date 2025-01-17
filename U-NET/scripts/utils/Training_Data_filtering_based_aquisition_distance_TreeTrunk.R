
library(reticulate)
# Setup Environment
reticulate::use_condaenv(condaenv = "tfr_terra",required = T)
.libPaths("/net/home/ssoltani/.conda/envs/tfr_terra/lib/R/library/")

library(tensorflow)
library(tfdatasets)
library(keras)
library(purrr)
library(gtools)
library(tidyverse)
n_bands=3
xres = 256L
yres = 256L



# GPU Configuration
gpu_indices <- c(1, 2)
gpu_devices <- tf$config$list_physical_devices()[gpu_indices]
tf$config$set_visible_devices(gpu_devices)

gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)


#put the mask as it is
base_dir <- "/mnt/gsdata/projects/bigplantsens/2_UNET_on_Flora_Mask/4_F_Japonica/data/"


#list the species folder
maskdir <- list.dirs(path = paste0(base_dir,"mask" ),recursive = FALSE)

#read all files
iNat_img <- as.data.frame(matrix(nrow = 1,ncol = 2))[-1,]
# #give columns name
colnames(iNat_img) <- c("msk","ref")
#
for(g in 1:length(maskdir)){
  images <- mixedsort(list.files(maskdir[g],full.names = T, pattern = ".png", recursive = T))
  ref <- rep(as.integer(g),length(images))
  findata <- tibble(msk=images,ref=ref)
  iNat_img <- rbind(iNat_img,findata)
}
#
#
# ####filter out the data
path_msk <- iNat_img$msk

path_img <- gsub("png$", "jpg", path_msk, ignore.case = TRUE)
path_img <- gsub("mask_", "",ignore.case = F, path_img)
path_img <- gsub("_mask", "",ignore.case = F, path_img)
path_img <- gsub("mask", "image", path_img)
path_img <- tibble(img=path_img)

#path_img <- "/mnt/gsdata/users/soltani/Workshop_home_fromSSD2/Workshop_home/2_Unet_on_flora_mask/2_MyDiv/data/Labeled_data_seprated_in_Folder/image"
#path_img <- list.files(path_img, pattern = "jpg",full.names = T,recursive = T) 
#final_data <- tibble(img=path_img)
create_dataset_data_filter <- function(data,
                           train, # logical. TRUE for augmentation of training data
                           batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                           epochs,
                           shuffle, # logical. default TRUE, set FALSE for test data
                           dataset_size){ # numeric. number of samples per epoch the model will be trained on
  
  
  # data1 <- data %>% group_by(ref) %>% sample_n(nrow(filter(data,data$ref==0))) %>% ungroup()
  # data2 <- data1[,1]
  # ref <- to_categorical(unlist(as.list(data1[,2])))
  # data <- tibble(data2, ref)
  if(shuffle){
    #sample subset of data in each epoch
    dataset = data %>%  
      tensor_slices_dataset() %>%
      
      dataset_shuffle(buffer_size = length(data$img), reshuffle_each_iteration = TRUE)
  } else {
    dataset = data %>%
      tensor_slices_dataset()
  }
  
  
  dataset = dataset %>%
    dataset_map(~.x %>% purrr::list_modify( # read files and decode png
      #img = tf$image$decode_png(tf$io$read_file(.x$img), channels = no_bands)
      img = tf$image$decode_jpeg(tf$io$read_file(.x$img)
                                 , channels = n_bands
                                 #, ratio = down_ratio
                                 , try_recover_truncated = TRUE
                                 , acceptable_fraction=0.5
      ) %>%
        tf$cast(dtype = tf$float32) %>%  
        tf$math$divide(255) %>% 
        #tf$image$convert_image_dtype(dtype = tf$float32) %>%
        tf$keras$preprocessing$image$smart_resize(size=c(xres, yres))))
 
  dataset = dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    #dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
    dataset_prefetch_to_device(device = "/gpu:0", buffer_size =tf$data$experimental$AUTOTUNE)
}



#take the data to the pipeline
#take the data to the pipeline
all_imgs <- create_dataset_data_filter(data =path_img,train = FALSE,batch = 1,shuffle = FALSE,dataset_size = length(path_img) )



###################################Distance prediction and denormalize
#load the trained model
model_path <- "/mnt/gsdata/projects/bigplantsens/00_Shared_data_model/00_checkpoints/00_Angle_Dist_stam_filter_models/"
Dist_model <- load_model_hdf5(paste0(model_path, "Log_transform_Distweights.49-0.01.hdf5"))

#Distance predictions
Dist_imgs_pred <- predict(object = Dist_model,x=all_imgs)


#denormalize the predictions
#logtransformation
minofdata <- -2.302585
maxofdata <- 5.010635
##############################normal transformation
# minofdata <- 0.1
# maxofdata <- 150

#denormalize function
denormalize <- function(x,minofdata,maxofdata) {
  x*(maxofdata-minofdata) + minofdata
}

Dist_pred_denormalized <- exp(denormalize(Dist_imgs_pred,minofdata,maxofdata))

###########################################put the images, ref, angle and distance prediction together


###model for stam no stam 
stam_model <- load_model_hdf5(paste0(model_path, "stam_no_stam_weights.39-0.00.hdf5"))
stam_no_stam <- as.array(k_argmax(predict(object = stam_model,x=all_imgs)))



############################################sample the number of images to combine with PlanNet data



#############Note important: This workflow is designed for 12 class classification procedure the last class, class12 belong to NAs and everything else
#it is defined where we export images.


#allimg_shap =  "/net/home/ssoltani/00 Workshop/04 myDive_tree_spec/01 Orthoimages/"




library(png)

library(stringr)
library(foreach)
library(doParallel)
cores <- 40#detectCores()-20

cl <- makePSOCKcluster(cores)
registerDoParallel(cl)




all_msks <- tibble(msk=path_msk)


# Assuming 'all_msks' is a data.frame with image file paths in 'msk' column
pixel_counts_list <- foreach(i = seq_len(nrow(all_msks)), .packages = c("png", "stringr"), .combine = rbind, .inorder = TRUE) %dopar% {
  tryCatch({
    # Read the PNG file and scale the values to 0-255
    png_data <- readPNG(all_msks$msk[i]) * 255
    
    # Count the pixels by value
    image_value_counts <- table(as.vector(png_data))
    
    # Calculate the total number of pixels
    total_pixels <- sum(image_value_counts)
    
    # Assuming background pixels are represented by 10 and scaled appropriately
    # Calculate the number of foreground pixels (all pixels that are not background)
    foreground_pixels <- total_pixels - image_value_counts['120']
    
    # Calculate foreground percentage
    foreground_percentage <- round(unname((foreground_pixels / total_pixels) * 100), digits = 2)
    
    # Prepare the pixel count data frame for this image
    pixel_count <- data.frame(matrix(data = NA, nrow = 1, ncol = 2))
    colnames(pixel_count) <- c("image", "px_count%")
    pixel_count[1, 1] <- paste0("mydiv_", sprintf("%07d", i), ".png")
    pixel_count[1, 2] <- as.numeric(foreground_percentage) # Assign pixel counts
    
    # Return the pixel count data frame
    pixel_count
  }, error = function(e) {
    # Handle the error by returning a data frame with NA values
    pixel_count <- data.frame(matrix(data = NA, nrow = 1, ncol = 2))
    colnames(pixel_count) <- c("image", "px_count%")
    pixel_count[1, 1] <- paste0("mydiv_", sprintf("%07d", i), ".png")
    pixel_count[1, 2] <- NA
    pixel_count
  })
}


###join the data
all_imgs_pred_join <- tibble(path_img,mask=path_msk,dist= Dist_pred_denormalized[,1],stam_nostam=stam_no_stam, pixel_percent = pixel_counts_list$`px_count%`,ref =iNat_img$ref )#,angle=angle_pred_denormalized[,1]

write.csv(all_imgs_pred_join, paste0(base_dir, "F_Japonica_plantNet_Distance_Stem_filter.csv"))


