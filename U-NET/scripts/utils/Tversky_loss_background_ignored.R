library(keras)
library(tensorflow)


# Tversky index function with adjustable alpha (used in Focal Tversky Loss)
tversky <- function(y_true, y_pred, alpha = 0.7, smooth = 1e-6) {
  # Flatten the true and predicted labels
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  
  # Mask out class 10 if needed
  valid_mask <- k_not_equal(y_true_f, 10)
  y_true_f <- tf$boolean_mask(y_true_f, valid_mask)
  y_pred_f <- tf$boolean_mask(y_pred_f, valid_mask)
  
  # Calculate true positives, false negatives, and false positives
  true_pos <- k_sum(y_true_f * y_pred_f)
  false_neg <- k_sum(y_true_f * (1 - y_pred_f))
  false_pos <- k_sum((1 - y_true_f) * y_pred_f)
  
  # Apply Tversky index formula
  tversky_index <- (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
  return(tversky_index)
}

# Focal Tversky Loss with gamma parameter
focal_tversky_loss <- function(y_true,y_pred, alpha = 0.3, gamma = 1.3, smooth = 1e-6) {
  tv <- tversky(y_true, y_pred, alpha = alpha, smooth = smooth)
  return(K$pow((1 - tv), gamma))
}

#We need them for accuracy metrics calaculation 

K <- backend()

# Dice coefficient function that ignores class 10
dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  # Flatten the true and predicted labels
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  
  # Find valid pixels (not class 10)
  valid_mask <- K$not_equal(y_true_f, 10)  # Create a mask to exclude class 10
  
  # Mask out class 10
  y_true_f <- tf$boolean_mask(y_true_f, valid_mask)
  y_pred_f <- tf$boolean_mask(y_pred_f, valid_mask)
  
  # Calculate Dice coefficient
  intersection <- K$sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) / (K$sum(y_true_f) + K$sum(y_pred_f) + smooth)
  
  return(result)
}




# model %>% compile(
#   optimizer = optimizer_adam(),
#   loss = tversky_loss(alpha=0.7, beta =0.3),  # Use only Tversky loss
#   metrics = custom_metric("dice_coef", dice_coef)  # Optionally include any metrics you want
# )
