#Customized loss function
# Loss function -----------------------------------------------------



# Customized dice loss
K <- backend()
dice_coef <- function(y_true, y_pred, smooth = 1.0) {
  y_true_f <- k_flatten(y_true)
  y_pred_f <- k_flatten(y_pred)
  
  #find NAs
  valids <- tf$math$is_finite(y_true_f)
  #mask NAs
  y_true_f <- tf$boolean_mask(y_true_f, valids)
  y_pred_f <- tf$boolean_mask(y_pred_f, valids)
  #
  intersection <- k_sum(y_true_f * y_pred_f)
  result <- (2 * intersection + smooth) / 
    (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
  return(result)
}

### customized dice losses

bce_dice_loss <- function(y_true, y_pred) {
  
  #modify the loss function 
  y_true <- k_flatten(y_true)
  y_pred <- k_flatten(y_pred)
  
  #find NAs
  valid2 <- tf$math$is_finite(y_true)
  #mask NAs
  y_true <- tf$boolean_mask(y_true, valid2)
  y_pred <- tf$boolean_mask(y_pred, valid2)
  
  result <- loss_binary_crossentropy(y_true, y_pred) +
    (1 - dice_coef(y_true, y_pred))
  return(result)
}




# K <- backend()
# dice_coef <- function(y_true, y_pred, smooth = 1.0) {
#   y_true_f <- k_flatten(y_true)
#   y_pred_f <- k_flatten(y_pred)
#   intersection <- k_sum(y_true_f * y_pred_f)
#   result <- (2 * intersection + smooth) / 
#     (k_sum(y_true_f) + k_sum(y_pred_f) + smooth)
#   return(result)
# }
# bce_dice_loss <- function(y_true, y_pred) {
#   result <- loss_binary_crossentropy(y_true, y_pred) +
#     (1 - dice_coef(y_true, y_pred))
#   return(result)
# }
