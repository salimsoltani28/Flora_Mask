

filepath = "/home/ms2487/workshop/Flora_Mask/data/1_data/image"
maskpath = "/home/ms2487/workshop/Flora_Mask/data/1_data/mask"

# list all files in the directory recursively with the pattern of jpg
all_imags_filtered = list.files(path = filepath, pattern = "jpg$", recursive = TRUE, full.names = TRUE)
mask_list = list.files(path = maskpath, pattern = "png$", recursive = TRUE, full.names = TRUE)
# now take the file list and only change

mask_list[1:5]

oldfile <- all_imags_filtered %>% unlist()

newpath <- paste0("/home/ms2487/workshop/Flora_Mask/data/Unet_trainingdata/mask/",1:length(mask_list),".png")

#get the basename of the file


file.copy(from = mask_list,to=newpath)

length(mask_list)

reverse_modify_paths <- function(paths) {
  modified_paths <- vector("list", length(paths))
  for (i in seq_along(paths)) {
    parts <- unlist(strsplit(paths[i], "/"))
    
    # Replace the first occurrence of "mask" with "image"
    parts[grep("mask", parts)[1]] <- "image"
    
    folder_name <- parts[length(parts) - 1]
    file_name <- parts[length(parts)]
    
    # Remove "_mask" from the folder name
    new_folder_name <- gsub("_mask$", "", folder_name)
    
    # Remove "mask_" prefix from the file name
    new_file_name <- gsub("^mask_", "", file_name)
    
    # Change the file extension from .png to .jpg
    new_file_name <- gsub("\\.png$", ".jpg", new_file_name)
    
    # Reconstruct the path with corrected parts
    modified_path <- paste(c(parts[1:(length(parts)-2)], new_folder_name, new_file_name), collapse = "/")
    
    modified_paths[[i]] <- modified_path
  }
  return(unlist(modified_paths)) # Convert list to vector for easier display/printing
}



# Get modified paths
modified_paths <- reverse_modify_paths(mask_list)
modified_paths[1:5]
# Display modified paths
print(modified_paths)

unlink("/home/ms2487/workshop/Flora_Mask/data/1_data/mask/",recursive = TRUE)
unlink("/home/ms2487/workshop/Flora_Mask/data/Unet_trainingdata/mask/",recursive = TRUE)
dir.create("/home/ms2487/workshop/Flora_Mask/data/Unet_trainingdata/image/")
dir.create("/home/ms2487/workshop/Flora_Mask/data/Unet_trainingdata/mask/")
