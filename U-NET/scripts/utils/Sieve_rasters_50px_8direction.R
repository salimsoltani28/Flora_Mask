# library(terra)
# library(foreach)
# library(doParallel)

# # Read command line arguments
# args <- commandArgs(trailingOnly = TRUE)

# message("Executing the Sieve vote function on the:", paste(args))
# predpath <- args

# allimages <- list.files(predpath, pattern = ".tif", recursive = FALSE, full.names = TRUE)

# # Set up parallel cluster
# cores <- 2  # Adjust based on system
# cl <- makePSOCKcluster(cores)
# registerDoParallel(cl)

# no_pix <- 2048

# # Parallel processing
# foreach(i = 1:length(allimages), .packages = "terra", .inorder = TRUE) %dopar% {
#   # Load raster within the loop
#   raster_file <- rast(allimages[i])
  
#   # Perform the sieve operation
#   raster1 <- terra::sieve(raster_file, threshold = no_pix, directions = 8)
#   raster1[raster1 < 0] <- NA
  
#   # Save output
#   output_file <- paste0(predpath, "/", i, "_5kpixel_final_pred_majorityvote_SIEVED_", no_pix, "PX.tif")
#   terra::writeRaster(raster1, output_file, overwrite = TRUE)
# }

# stopCluster(cl)
# message("Processing complete.")


library(raster)
library(terra)
library(foreach)
library(doParallel)

# Read command line arguments
args <- commandArgs(trailingOnly = TRUE)

message("Executing the Clump and Sieve function on the:", paste(args))
predpath <- args

allimages <- list.files(predpath, pattern = ".tif", recursive = FALSE, full.names = TRUE)

# Set up parallel cluster
cores <- 2  # Adjust based on your system
cl <- makePSOCKcluster(cores)
registerDoParallel(cl)

# Ensure required libraries are loaded on worker nodes
clusterEvalQ(cl, {
  library(raster)
  library(terra)
})

no_pix <- 32  # Minimum size threshold for regions to retain

# Parallel processing
foreach(i = 1:length(allimages), .packages = c("raster", "terra"), .inorder = TRUE) %dopar% {
  # Load raster within the loop
  raster_file <- raster(allimages[i])
  
  # Step 1: Clump operation - identify connected regions
  clumped_raster <- clump(raster_file, directions = 8)
  
  # Step 2: Filter clumps by size
  clump_sizes <- freq(clumped_raster)  # Calculate clump sizes
  large_clumps <- clump_sizes$value[clump_sizes$count >= no_pix]  # IDs to keep
  
  # Mask smaller clumps
  filtered_raster <- clumped_raster
  filtered_raster[!clumped_raster[] %in% large_clumps] <- NA
  
  # Save output
  output_file <- paste0(predpath, "/", i, "_final_pred_clump_filtered_", no_pix, "PX.tif")
  writeRaster(filtered_raster, output_file, overwrite = TRUE)
}

stopCluster(cl)
message("Processing complete.")
