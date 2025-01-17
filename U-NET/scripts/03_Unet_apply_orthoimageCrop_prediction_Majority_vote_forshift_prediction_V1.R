#.libPaths("/net/home/ssoltani/R/x86_64-pc-linux-gnu-library/4.3")
library(tidyverse)
library(raster)
library(snow)
library(raster)
library(terra)
#ls('package:raster')

# Read command line arguments
args <- commandArgs(trailingOnly = TRUE)


message("Eexecuing the majority vote function on the :", paste(args))
predpath <- args

allimages <- list.files(predpath, pattern = ".tif", full.names = T)
save_path = paste0(predpath,"/majority_vote/")
dir.create(save_path)


for (i in seq(1, length(allimages), 3)) {
  rasterstack <-  raster::stack(allimages[i:(i+2)])
  # do something with chunk
  # majority vote
  beginCluster(n=10)
  finalPrediction = clusterR(rasterstack, calc, args = list(modal, na.rm = T, ties = "random"))
  endCluster()
  #crop the raster around the margin by 512 pix
  div_factor <- 8192* res(finalPrediction)[1]
  finalPrediction <- crop(finalPrediction, extent(finalPrediction)/div_factor)
  
  writeRaster(finalPrediction, paste0(save_path,i, "_8kcrop_final_pred_majorityvote_test.tif"))
}

#Accuracy assessment
evaluation_script = "/mnt/gsdata/projects/bigplantsens/2_UNET_on_Flora_Mask/01_MyDiv/evaluation/Acuracy_assesment_CNNjulyOrtho_segment_MyDiv_tree_species_perTile.R"

# Trigger evaluation script
evaluation_input_path <- save_path  # Use save_path or any required path as the input

if (file.exists(evaluation_script)) {
    system(paste("Rscript", shQuote(evaluation_script), shQuote(evaluation_input_path)))
    message("Triggered evaluation script with input path:", evaluation_input_path)
} else {
    message("Evaluation script not found:", evaluation_script)
}
