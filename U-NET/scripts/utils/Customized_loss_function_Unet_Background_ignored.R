#class to be ignored
class_ignored = class_ignored

K <- backend()

# Dice coefficient function that ignores class 10
dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  # Flatten the true and predicted labels
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  
  # Find valid pixels (not class 10)
  valid_mask <- K$not_equal(y_true_f, class_ignored)  # Create a mask to exclude class 10
  
  # Mask out class 10
  y_true_f <- tf$boolean_mask(y_true_f, valid_mask)
  y_pred_f <- tf$boolean_mask(y_pred_f, valid_mask)
  
  # Calculate Dice coefficient
  intersection <- K$sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) / (K$sum(y_true_f) + K$sum(y_pred_f) + smooth)
  
  return(result)
}

# Customized binary cross-entropy + dice loss function that ignores class 10
bce_dice_loss <- function(y_true, y_pred) {
  # Flatten the true and predicted labels
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  
  # Find valid pixels (not class 10)
  valid_mask <- K$not_equal(y_true_f, class_ignored)  # Create a mask to exclude class 10
  
  # Mask out class 10
  y_true_f <- tf$boolean_mask(y_true_f, valid_mask)
  y_pred_f <- tf$boolean_mask(y_pred_f, valid_mask)
  
  # Calculate binary cross-entropy and dice loss
  bce_loss <- loss_binary_crossentropy(y_true_f, y_pred_f)
  dice_loss <- 1 - dice_coef(y_true_f, y_pred_f)
  
  # Combine BCE and dice loss
  result <- bce_loss + dice_loss
  
  return(result)
}

