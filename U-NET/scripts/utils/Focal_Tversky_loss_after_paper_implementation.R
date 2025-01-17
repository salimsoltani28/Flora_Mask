library(keras)
library(tensorflow)

Ignored_class= Ignored_class
# Define the Tversky index function
tversky_index <- function(y_true, y_pred, alpha = 0.7, beta = 0.3, smooth = 1e-6) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)

  # Mask out class 10 if needed
  valid_mask <- k_not_equal(y_true_f, Ignored_class)
  y_true_f <- tf$boolean_mask(y_true_f, valid_mask)
  y_pred_f <- tf$boolean_mask(y_pred_f, valid_mask)
  
  
  # Calculate true positives, false negatives, and false positives
  true_pos <- k_sum(y_true_f * y_pred_f)
  false_neg <- k_sum(y_true_f * (1 - y_pred_f))
  false_pos <- k_sum((1 - y_true_f) * y_pred_f)
  
  tversky <- (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
  return(tversky)
}

# Define the Focal Tversky Loss function
focal_tversky_loss <- function(alpha = 0.5, beta = 0.5, gamma = 1.5, smooth = 1e-6) {
  function(y_true, y_pred) {
    tversky <- tversky_index(y_true, y_pred, alpha, beta, smooth)
    loss <- k_pow((1 - tversky), gamma)
    return(loss)
  }
}



K <- backend()


# Dice coefficient function that ignores class 10
dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  # Flatten the true and predicted labels
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  
  # Find valid pixels (not class 10)
  valid_mask <- K$not_equal(y_true_f, Ignored_class)  # Create a mask to exclude class 10
  
  # Mask out class 10
  y_true_f <- tf$boolean_mask(y_true_f, valid_mask)
  y_pred_f <- tf$boolean_mask(y_pred_f, valid_mask)
  
  # Calculate Dice coefficient
  intersection <- K$sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) / (K$sum(y_true_f) + K$sum(y_pred_f) + smooth)
  
  return(result)
}
