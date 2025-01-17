create_dataset <- function(data,
                           train, # logical. TRUE for augmentation of training data
                           batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                           epochs,
                           shuffle = TRUE, # logical. default TRUE, set FALSE for test data
                           dataset_size, # numeric. number of samples per epoch the model will be trained on
                           image_size , # Added this parameter for flexibility
                           no_classes) { # Added no_classes parameter for one_hot encoding
  require(tfdatasets)
  require(tensorflow)
  require(purrr)
  
  # Define resize size as a Tensor, adjusting based on new parameter
  resize_size <- tf$constant(c(image_size, image_size), dtype = 'int32')
  
  if(shuffle){
    dataset <- data %>%
      tensor_slices_dataset() %>%
      dataset_shuffle(buffer_size = length(data$img), reshuffle_each_iteration = TRUE)
  } else {
    dataset <- data %>%
      tensor_slices_dataset()
  } 
  
  # dataset <- dataset %>%
  #   dataset_map(~.x %>% list_modify( # read files, decode, and resize images and masks
  #     img = tf$image$decode_jpeg(tf$io$read_file(.x$img), channels = 3L) %>% # Assuming 3 channels for images
  #       tf$image$resize(size = resize_size), # Corrected resize call
  #     msk = tf$image$decode_png(tf$io$read_file(.x$msk), channels = 1L) %>%
  #       tf$image$resize(size = resize_size, method = "nearest") # Corrected resize call
  #   )) 
    
# Adding try-catch to handle missing or inaccessible files
    dataset <- dataset %>%
      dataset_map(~tryCatch({
        list_modify(.x,
                    img = tf$image$decode_jpeg(tf$io$read_file(.x$img), channels = 3L,
                    try_recover_truncated=TRUE,acceptable_fraction=0.5 ) %>%
                      tf$image$resize(size = resize_size),
                    msk = tf$image$decode_png(tf$io$read_file(.x$msk), channels = 1L) %>%
                      tf$image$resize(size = resize_size, method = "nearest")
        )
      }, error = function(e) {
        # Skip this example if there's an error
        NULL
      }),num_parallel_calls = tf$data$AUTOTUNE) %>%
      dataset_filter(function(x) !is.null(x))  # Remove NULL entries (skipped files)
  #end of the new code
    dataset <- dataset %>%
    dataset_map(~.x %>% list_modify(
      img = tf$image$convert_image_dtype(.x$img, dtype = tf$float32) %>%
        tf$math$divide(255.0) , #normalize the image value
      msk = tf$one_hot(tf$squeeze(tf$cast(.x$msk, tf$int32)), depth = as.integer(no_classes), dtype = tf$float32)%>%
        tf$squeeze()
    )) %>%
    dataset_map(~.x %>% list_modify(
      img = tf$reshape(.x$img, shape = c(image_size, image_size, 3L)),
      msk = tf$reshape(.x$msk, shape = c(image_size, image_size, no_classes))
    ))
  
  if(train) {
    dataset <- dataset %>%
      dataset_map(function(x) {
        # Generate random flip parameters using integers for minval and maxval
        flip_left_right <- tf$random$uniform(shape = list(), minval = 0L, maxval = 2L, dtype = tf$int32)
        flip_up_down <- tf$random$uniform(shape = list(), minval = 0L, maxval = 2L, dtype = tf$int32)
        
        # Apply conditional flips to both img and msk
        x$img <- tf$cond(flip_left_right > 0, function() tf$image$flip_left_right(x$img), function() x$img)
        x$msk <- tf$cond(flip_left_right > 0, function() tf$image$flip_left_right(x$msk), function() x$msk)
        x$img <- tf$cond(flip_up_down > 0, function() tf$image$flip_up_down(x$img), function() x$img)
        x$msk <- tf$cond(flip_up_down > 0, function() tf$image$flip_up_down(x$msk), function() x$msk)
        
        # Continue with your other transformations for img
        # Note: These additional transformations should not be applied to msk
        x$img <- x$img %>%
          tf$image$random_brightness(max_delta = 0.1) %>%
          tf$image$random_contrast(lower = 0.9, upper = 1.1) %>%
          tf$image$random_saturation(lower = 0.9, upper = 1.1) %>% 
          tf$image$random_hue(max_delta = 0.02) %>%
          tf$clip_by_value(0, 1)
        
        x
      }) %>%
      dataset_repeat(count = ceiling(epochs * dataset_size / batch))
  } 
  # else {
  #   dataset <- dataset %>%
  #     dataset_repeat(count = ceiling(epochs * dataset_size / batch))
  # }
  
  dataset <- dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    dataset_prefetch_to_device( paste0("/gpu:",gpu_index),buffer_size = tf$data$AUTOTUNE)
  
  return(dataset)
}
