# Set the command as a string
#Check the loss function before making the predictions

command <- "Rscript 02_Unet_apply_July_orthoimageCrop_prediction_V1.R EXP3.3_over0.2under20_stem_30percent_cls11_Ortho_backRplcd_ZMEXCLD_Adamw_gelu_img512_80/ 21 40 256" #this one was executed

# Execute the command
system(command)
