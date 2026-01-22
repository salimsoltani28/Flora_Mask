##Before starting the work, some sanity check on data and workflow 
## check if the ref data correspond to the images path 
## change image size accordingly 
## regression ? scale the ref labels 0,1
## Check explicitly if those training data Augmentation is relevent to you


# install_keras(tensorflow = "gpu")

library(magick)
require(keras)
library(tensorflow)
library(tfdatasets)
library(tidyverse)
library(tibble)
library(rsample)
library(countcolors)
library(reticulate)
require(gtools)
tf$compat$v1$set_random_seed(as.integer(28))
#tfe_enable_eager_execution(device_policy = "silent") # enables eager execution (run directly after library(tensorflow))
#tf$compat$v1$set_random_seed(as.integer(28)) # this is how to set seeds with tensorflow (to ensure reproduceability of code)

# set memory growth policy
gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
#gpu2 <- tf$config$experimental$get_visible_devices('GPU')[[2]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)

#tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE) #strategy <- tf$distribute$MirroredStrategy() # required for using multiple GPUs, uncomment both lines in case just one GPU is used strategy$num_replicas_in_sync


workdir = "/home/ssoltani/workshop/04 Training data optimization/02 Version2 labeled data_balanced_good_4_Distance/"
outdir = "/home/ssoltani/workshop/04 Training data optimization/02 Version2 labeled data_balanced_good_4_Distance/output"


path_img1= paste0(workdir,"01 Fagus sylvatica")
path_img2 = paste0(workdir, "02 Picea abies")
path_img3 = paste0(workdir, "03 Quercus spp")
path_img4 = paste0(workdir, "04 Carpinus_betulus")
path_img5 = paste0(workdir,"05 Pseudotsuga menziesii")
path_img6 = paste0(workdir, "06 Pinus sylvestris")
path_img7 = paste0(workdir,"07 Acer spp")
path_img8 = paste0(workdir,"08 Betula pendula")
path_img9 = paste0(workdir, "09 Tilia spp")
path_img10 = paste0(workdir, "10 Fraxinus excelsior")
path_img11 = paste0(workdir, "11 Larix decidua")
path_img12 = paste0(workdir, "12  Abies_alba")
path_img13 = paste0(workdir, "13 Forest floor")
path_img14 = paste0(workdir, "14 Angle_Distance")
path_img15 = paste0(workdir, "15 Distance For Balnce")
path_img16 = paste0(workdir, "16 Extra_train_Distance")
path_img17 = paste0(workdir, "17 Dist4_balance")

#set the image size
xres = 256L
yres = 256L
n_bands = 3L

setwd(workdir)



################################################### Loading Data


# list all img  data
path_img1 = mixedsort(list.files(path_img1, full.names = T,pattern = ".JPG", recursive = T))
path_img2 = mixedsort(list.files(path_img2, full.names = T, pattern = ".JPG", recursive = T))
path_img3 = mixedsort(list.files(path_img3, full.names = T, pattern = ".JPG", recursive = T))
path_img4 = mixedsort(list.files(path_img4, full.names = T, pattern = ".JPG", recursive = T))
path_img5 = mixedsort(list.files(path_img5, full.names = T, pattern = ".JPG", recursive = T))
path_img6 = mixedsort(list.files(path_img6, full.names = T, pattern = ".JPG", recursive = T))
path_img7 = mixedsort(list.files(path_img7, full.names = T, pattern = ".JPG", recursive = T)) 
path_img8 = mixedsort(list.files(path_img8, full.names = T, pattern = ".JPG", recursive = T)) 
path_img9 = mixedsort(list.files(path_img9, full.names = T, pattern = ".JPG", recursive = T))
path_img10 = mixedsort(list.files(path_img10, full.names = T, pattern = ".JPG", recursive = T))
path_img11 = mixedsort(list.files(path_img11, full.names = T, pattern = ".JPG", recursive = T))
path_img12 = mixedsort(list.files(path_img12, full.names = T, pattern = ".JPG", recursive = T))
path_img13 = mixedsort(list.files(path_img13, full.names = T, pattern = ".JPG", recursive = T))
path_img14 = mixedsort(list.files(path_img14, full.names = T, pattern = ".jpg", recursive = T))
path_img15 = mixedsort(list.files(path_img15, full.names = T, pattern = ".jpg", recursive = T))
path_img16 = mixedsort(list.files(path_img16, full.names = T, pattern = ".JPG", recursive = T))
path_img17 = mixedsort(list.files(path_img17, full.names = T, pattern = ".JPG", recursive = T))

path_img = c(path_img1, path_img2,path_img3,path_img4,path_img5,path_img6,path_img7,path_img8,path_img9, path_img10, path_img11, path_img12,path_img13,path_img14,path_img15,path_img16,path_img17)





################################################# Loading the refrence data
#******important***see  if the ref data correspond to the image paths

spec1 <- read.csv(paste0(workdir,"01 Fagus sylvatica/Fagus_sylvatica_list _Modified.csv"))[,c(2,3)]
spec2 <- read.csv(paste0(workdir,"02 Picea abies/Picea_abies_modified.csv"))[,c(2,3)]
spec3 <- read.csv(paste0(workdir,"03 Quercus spp/Quercus_spp_Modified.csv"))[,c(2,3)]
spec4 <- read.csv(paste0(workdir,"04 Carpinus_betulus/Carpinus_betulus_Modified.csv"))[,c(2,3)]
spec5 <- read.csv(paste0(workdir,"05 Pseudotsuga menziesii/Pseudotsuga_menziesii_Modified.csv"))[,c(2,3)]
spec6 <- read.csv(paste0(workdir,"06 Pinus sylvestris/Pinus_sylvestris_list_Modified.csv"))[,c(2,3)]
spec7 <- read.csv(paste0(workdir,"07 Acer spp/Acer_spp_list - edited.csv"))[,c(2,3)]
spec8 <- read.csv(paste0(workdir,"08 Betula pendula/Betula_pendula_modified.csv"))[,c(2,3)]
spec9 <- read.csv(paste0(workdir,"09 Tilia spp/Tilia_spp_modified.csv"))[,c(2,3)]
spec10 <- read.csv(paste0(workdir,"10 Fraxinus excelsior/Fraxinus excelsior_list_Modified.csv"))[,c(2,3)]
spec11 <- read.csv(paste0(workdir,"11 Larix decidua/Larix_decidua_modified.csv"))[,c(2,3)]
spec12 <- read.csv(paste0(workdir,"12  Abies_alba/Abies_alba_modified.csv"))[,c(2,3)]
spec13 <- read.csv(paste0(workdir,"13 Forest floor/forest_floor_modified.csv"))[,c(2,3)]
spec14 <- read.csv(paste0(workdir,"14 Angle_Distance/Angle_Distance_prop.csv"))[,c(2,3)]
spec15 <- read.csv(paste0(workdir,"15 Distance For Balnce/01 Final_Balance_Distance.csv"))[,c(2,3)]
spec16 <- read.csv(paste0(workdir,"16 Extra_train_Distance/Final_distance.csv"))[,c(2,3)]
spec17 <- read.csv(paste0(workdir,"17 Dist4_balance/Dist_4_balance_modified.csv"))[,c(2,3)]

ref1 = rbind(spec1,spec2,spec3,spec4,spec5,spec6,spec7,spec8, spec9, spec10, spec11, spec12, spec13,spec14,spec15,spec16,spec17)
ref = rbind(spec1,spec2,spec3,spec4,spec5,spec6,spec7,spec8, spec9, spec10, spec11, spec12, spec13,spec14,spec15,spec16,spec17)

hist(ref$Angle)
hist(ref$Distance)

# hist(Distorg)
#normlize function 
range01 <- function(x){(x-min(x))/(max(x)-min(x))}

Distance <- range01(ref$Distance)
Angle <- range01(ref$Angle)

ref <- Distance

######################################################################################################Filter the datastet
# 
# Distance <- ref1$Distance
# Filtered_ref <- tibble(path_img,Distance)
# labelfreq <- Filtered_ref %>% group_by(Distance) %>% summarise(numberofData=n())
# #because of image scarcity in far distances we shrink all distance above 100 to 150 to see the effect 
# Filtered_ref$Distance[Filtered_ref$Distance<1] <- 1
# ##filter data for each group
# finaldata <- Filtered_ref %>%
#   group_by( Distance) %>%
#   filter(n() <50)
# # #
# # finaldata <- Filtered_ref %>% str_replace(Distance, .>130, 130)
# #
# # hist(finaldata$Distance)
# #
#  Filteredref <-Filtered_ref
# # ggplot(Filteredref) +
# #   geom_histogram(mapping = aes(x = Distance), binwidth = 0.5)
# #
# ref <- range01(Filteredref$Distance)
# path_img <- Filteredref$path_img
###################################################training and test

testIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/10), replace = F)
test_img = path_img[testIdx]
save(test_img, file = paste0(outdir, "test_img.RData"), overwrite = T)
test_ref = ref[testIdx]
save(test_ref, file = paste0(outdir, "test_ref.RData"), overwrite = T)
# split training and validation data
path_img = path_img[-testIdx]
ref = ref[-testIdx]
valIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/5), replace = F)
val_img = path_img[valIdx]
val_ref = ref[valIdx]
train_img = path_img[-valIdx];
train_ref = ref[-valIdx]


val_data = tibble(img = val_img, val_ref)
train_data = tibble(img = train_img, train_ref)

# ref1 <- train_data$train_ref
# 
# refn <- ref1+k_random_normal(shape = list(1, 1),  stddev = 0.05, dtype = tf$float32)

# ##################################################################convert it back to label to be feeded in the pipeline
# finalref <- as.array(refn)
# 
# train_data = tibble(img = train_img, finalref)

########################################################## tfdatasets input pipeline
create_dataset <- function(data,
                           train, # logical. TRUE for augmentation of training data
                           batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                           epochs,
                           shuffle, # logical. default TRUE, set FALSE for test data
                           dataset_size){ # numeric. number of samples per epoch the model will be trained on
  if(shuffle){
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
        tf$image$convert_image_dtype(dtype = tf$float32) %>%
        tf$keras$preprocessing$image$smart_resize(size=c(xres, yres))))
  
  #old resize function for tensorflow 2.2
  #     tf$image$resize(preserve_aspect_ratio = TRUE, size = as.integer(c(ceiling(xres*2.1)
  #                                                                       ,ceiling(yres*2.1)))
  #     ) %>%
  #     tf$image$resize_with_crop_or_pad(target_height = yres, target_width = xres)
  #   
  # ))  #, num_parallel_calls = tf$data$experimental$AUTOTUNE)
  
  
  # if(train) {
  # 
  #   dataset = dataset %>%
  #     dataset_repeat(count = ceiling(epochs *(dataset_size/length(train_data$img))))}

  if(train) {


    #data augmentation
    dataset = dataset %>%
      dataset_map(~.x %>% purrr::list_modify(
        img= tf$add(.x$img,k_random_normal(shape = list(1, 1),  stddev = 0.05, dtype = tf$float32)) %>% 
        tf$image$random_flip_left_right() %>%
          tf$image$random_brightness(max_delta = 0.1, seed = 1L) %>%
          tf$image$random_contrast(lower = 0.9, upper = 1.1) %>%
          tf$image$random_saturation(lower = 0.9, upper = 1.1) %>% # requires 3 chnl -> with useDSM chnl = 4
          tf$clip_by_value(0, 1) # clip the values into [0,1] range.

      )) %>% #,num_parallel_calls = tf$data$experimental$AUTOTUNE
      #),num_parallel_calls = NULL) %>%
  dataset_repeat(count = ceiling(epochs *(dataset_size/length(train_data$img))))}
  dataset = dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    #dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
    dataset_prefetch_to_device(device = "/gpu:0", buffer_size =tf$data$experimental$AUTOTUNE)
}


#######################################################################################Parameters###################################################


batch_size <-10 # 12 (multi gpu, 512 a 2cm --> rstudio freeze) 
n_epochs <- 50 
dataset_size <- length(train_data$img) # if ortho is combined with DSM = 4 (RGB + DSM), if not = 3 (RGB)

training_dataset <- create_dataset(train_data, train = TRUE, batch = batch_size, epochs = n_epochs, dataset_size = dataset_size,shuffle = TRUE) 
validation_dataset <- create_dataset(val_data, train = FALSE, batch = batch_size, epochs = n_epochs,shuffle = TRUE)

# with the following lines you can test if your input pipeline produces meaningful tensors. You can also use as.raster, etc... to visualize the frames.
dataset_iter = reticulate::as_iterator(training_dataset)
example = dataset_iter %>% reticulate::iter_next() 
example
plotArrayAsImage(as.array(example[[1]][1,,,]))
example[[2]][1,]


dataset_iter = reticulate::as_iterator(validation_dataset)
example = dataset_iter %>% reticulate::iter_next() 
example
plotArrayAsImage(as.array(example[[1]][2,,,]))
example[[2]][1,]


########################################################## Defining Model

# model <- keras_model_sequential() %>%
#   layer_separable_conv_2d(filters = 32, kernel_size = 3,
#                           activation = "relu",
#                           input_shape = c(xres, yres, n_bands)) %>%
#   layer_separable_conv_2d(filters = 64, kernel_size = 3,
#                           activation = "relu") %>%
#   
#   layer_global_average_pooling_2d() %>%
#   layer_dropout(rate = 0.5) %>%
#   layer_dense(units = 512, activation = "relu") %>%
#   layer_dense(units = 1, activation = "sigmoid")
base_model <- application_densenet121( include_top = FALSE, input_shape = c(xres, yres, n_bands))

# add our custom layers
predictions <- base_model$output %>%
  layer_global_average_pooling_2d() %>% 
  #layer_gaussian_noise(stddev=0.05) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 512L, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512L, activation = 'relu') %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 256L, activation = 'relu') %>% #256
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 128L, activation = 'relu') %>%  #32
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 64L, activation = 'relu') %>% 
  layer_dense(units = 1L, activation = 'sigmoid') #test linear vs sigmoid

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)




###############################################################compile parameters
model %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.0001), # test different learning rate
  loss = "mse", #Mean sqa error
  metrics = c("mae")
)

################################################################### Model fitting 


checkpoint_dir <- paste0(outdir, "checkpoints_shrined150_Densenet121_4_26")
unlink(checkpoint_dir, recursive = TRUE)
dir.create(checkpoint_dir, recursive = TRUE)
filepath = file.path(checkpoint_dir, 
                     "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

cp_callback <- callback_model_checkpoint(filepath = filepath,
                                         monitor = "val_loss",
                                         save_weights_only = FALSE,
                                         save_best_only = TRUE,
                                         verbose = 1,
                                         mode = "auto",
                                         save_freq = "epoch")

history <- model %>% fit(x = training_dataset,
                         epochs = n_epochs,
                         steps_per_epoch = dataset_size/batch_size,
                         callbacks = list(cp_callback, 
                                          callback_terminate_on_naan()),
                         validation_data = validation_dataset)



########################################################################### prediction of training data


#####################
#### EVALUTATION ####
#####################
#outdir = "results/"

checkpoint_dir <- paste0( outdir, "/checkpoints_training_opt/")
load(paste0(outdir, "test_img.RData"))
load(paste0(outdir, "test_ref.RData"))
testdata = tibble(img = test_img,
                  ref = test_ref)

              
test_dataset <- create_dataset(testdata, train = FALSE, batch = 1, 
                               shuffle = FALSE)

model = load_model_hdf5('weights.06-0.04.hdf5', compile = TRUE)

eval <- evaluate(object = model, x = test_dataset)
eval
test_pred = predict(model, test_dataset)
#################################################
#min and max of original data
Dist4Normal <- ref1$Distance#as.numeric(gsub(",", ".", gsub("\\.", "", reforig$Distance)))

minofdata <- min(Dist4Normal)
maxofdata <- max(Dist4Normal)

#function
denormalize <- function(x,minofdata,maxofdata) {
  x*(maxofdata-minofdata) + minofdata
}


test_pred <-denormalize(test_pred,minofdata,maxofdata )
testref_de <- denormalize(testdata$ref,minofdata,maxofdata)
###################################################################predict on test dataset



plot(testref_de,test_pred,ylim=c(0,150), xlim=c(0,150))

cor.test(testref_de,test_pred)

#prediction on the test data
#function r2 calculation
cor.test(testref_de, test_pred)$estimate^2

#training_datadir ="/home/ssoltani/workshop/10kdata"



#####################predict on the training data
test_training_dataset <- create_dataset(train_data, train = FALSE, batch = 1, 
                               shuffle = FALSE)
test_train_pred = predict(model, test_training_dataset)

#denormalize
test_train_pred <-denormalize(test_train_pred,minofdata,maxofdata )
test_train_ref <- denormalize(train_data$train_ref,minofdata,maxofdata)

plot(test_train_ref, test_train_pred,ylim=c(0,150), xlim=c(0,150))

cor.test(test_train_ref, test_train_pred)
cor.test(test_train_ref, test_train_pred)$estimate^2


###############################################################################if your model is accurate, it is time to deploy it on big data
training_datadir ="/home/ssoltani/workshop/10kdata"
train_img1= paste0(training_datadir,"/01 Fagus_sylvatica")
train_img2 = paste0(training_datadir,"/02 Picea_abies")
train_img3 = paste0(training_datadir,"/03 Quercus spp")
train_img4 = paste0(training_datadir,"/04 Carpinus betulus")
train_img5 = paste0(training_datadir,"/05 Pseudotsuga_menziesii")
train_img6 = paste0(training_datadir,"/06 Pinus sylvestris")
train_img7 = paste0(training_datadir,"/07 Acer spp")
train_img8 = paste0(training_datadir,"/08 Betula_pendula")
train_img9 = paste0(training_datadir,"/09 Tilia_spp")
train_img10 = paste0(training_datadir,"/10 Fraxinus excelsior")
train_img11 = paste0(training_datadir,"/11 Larix_decidua")
train_img12 = paste0(training_datadir,"/12 Abies_alba")
train_img13 = paste0(training_datadir,"/13 forest floor")



############################################

# list all img  data

train_img1 = list.files(train_img1, full.names = T,pattern = ".jpg", recursive = T)
train_img2 = list.files(train_img2, full.names = T, pattern = ".jpg", recursive = T)
train_img3 = list.files(train_img3, full.names = T, pattern = ".jpg", recursive = T)
train_img4 = list.files(train_img4, full.names = T, pattern = ".jpg", recursive = T)
train_img5 = list.files(train_img5, full.names = T, pattern = ".jpg", recursive = T)
train_img6 = list.files(train_img6, full.names = T, pattern = ".jpg", recursive = T) 
train_img7 = list.files(train_img7, full.names = T, pattern = ".jpg", recursive = T) 
train_img8 = list.files(train_img8, full.names = T, pattern = ".jpg", recursive = T) 
train_img9 = list.files(train_img9, full.names = T, pattern = ".jpg", recursive = T)
train_img10 = list.files(train_img10, full.names = T, pattern = ".jpg", recursive = T)
train_img11 = list.files(train_img11, full.names = T, pattern = ".jpg", recursive = T)
train_img12 = list.files(train_img12, full.names = T, pattern = ".jpg", recursive = T)
train_img13 = list.files(train_img13, full.names = T, pattern = ".JPG", recursive = T)
path_img = c(train_img1, train_img2,train_img3,train_img4,train_img5,train_img6,train_img7,train_img8, train_img9, train_img10,train_img11,train_img12, train_img13)


trainingdata_input = tibble(img = path_img)
train_dataset <- create_dataset(trainingdata_input, train = FALSE, batch = 1, 
                               shuffle = FALSE)
train_opt_pred = predict(model, train_dataset)
save(train_opt_pred,"train_opt_pred.RData")

sum(train_opt_pred[10,])

#######################################################denormalize
#min and max of original data
Dist4Normal <- ref1$Distance#as.numeric(gsub(",", ".", gsub("\\.", "", reforig$Distance)))

minofdata <- min(Dist4Normal)
maxofdata <- max(Dist4Normal)

#function
denormalize <- function(x,minofdata,maxofdata) {
  x*(maxofdata-minofdata) + minofdata
}


Pred_dist_denormalized <-denormalize(test_pred,minofdata,maxofdata )
finalnatija <- tibble(path_img16,Pred_dist_denormalized)
#####Angle Demoralization

hist(Pred_denormalized)
min(Pred_denormalized)
max(Pred_denormalized)

write.csv(Pred_dist_denormalized, "Pred_dist_denormalized_finalprediction.csv")

# #############################################################Test if normalize and denormalize function output the original number
# range01 <- function(x){(x-min(x))/(max(x)-min(x))}
# 
# #change from comma to point for decimal 
# 
# normalztest <- range01(reforig$Angle)
# 
# #min and max of original data
# 
# minofdata <- min(Dist4Normal)
# maxofdata <- max(Dist4Normal)
# 
# #function
# denormalize <- function(x,minofdata,maxofdata) {
#   x*(maxofdata-minofdata) + minofdata
# }
# 
# 
# denormlized <-denormalize(normalztest,minofdata,maxofdata )
# 
# #hist 
# hist(denormlized)
# 
# #hist of original data
# hist(reforig$Angle)
# 
# 
# ######if you want put them in one dataframe and you can check some the data visually
# newdata4chceck <- data.frame(newdata,reforig$Angle )
