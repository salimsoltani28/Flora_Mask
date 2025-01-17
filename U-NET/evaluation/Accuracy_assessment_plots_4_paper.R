
library(reticulate)
require(raster)
require(rgdal)
library(tidyverse)
library(rgeos)
library(ggplot2)
library(gtools)
library(gridExtra)
library(dplyr)
library(ROCR)
library(MLmetrics)
library(sf)
library(doParallel)
library(foreach)
library(terra)


data_path <- "Z:/projects/bigplantsens/2_UNET_on_Flora_Mask/01_MyDiv/Pred_output/Ortho_July_/EXP3_over0.2under20_stem_30percent_cls11_Ortho_backRplcd_ZoomEXCLD_img512_80/Majority_vote/"

###saved
#data_with_saved <- Acc_val
#change for plots
Acc_val <- paste0(data_path,"meanTreespecF1_0.25_F1gr0.5_1_per_tile_forallplots.csv") %>% read_csv()







# Here enter if comparing between the orthos are true


PerclasF1 <- Acc_val 

#add the plot type
# Function to replace patterns with values
replace_pattern <- function(data) {
  # Extract numeric values after the letters
  values <- gsub("^.*[A-Za-z]+([0-9]+)$", "\\1", data)
  
  # Convert the extracted values to numeric
  values <- as.numeric(values)
  
  return(values)
}

#complete data
PerclasF1 <- PerclasF1 %>% 
  mutate(Plot_type=replace_pattern(Plot_type)) %>% filter(Plot_type==1)
# Generate and save the plot for the complete data

# Compute mean F1 score for each list of F1 scores
No_class_F1_high <- PerclasF1 %>% group_by(Species) %>% 
  summarize(meanf1=mean(F1)) %>% 
  filter(Species!="Grass" & meanf1>0.5) %>% count()

# Compute mean F1 score for each list of F1 scores
Treeclass_mean <- PerclasF1 %>% group_by(Species) %>% 
  summarize(meanf1=mean(F1)) %>% 
  filter(Species!="Grass" ) %>% pull(meanf1) %>% mean() %>% 
  round(digits = 2)

# # Replace the values
# PerclasF1$Species <- gsub("\\.", " ", PerclasF1$Species)
# PerclasF1$Species <- gsub("([^\\.])$", "\\1.", PerclasF1$Species)
# PerclasF1$Species <- ifelse(PerclasF1$Species == "Grass.", "Grass", PerclasF1$Species)



# Alternatively, in a more compact form:
PerclasF1$Species <- ifelse(gsub("([^\\.])$", "\\1.", gsub("\\.", " ", PerclasF1$Species)) == "Grass.", "Grass", gsub("([^\\.])$", "\\1.", gsub("\\.", " ", PerclasF1$Species)))






# Assume that your raster values are integers from 1 to 11
# (Adjust if your raster has a different range of values)
#color_palette <- setNames(colors, 1:11)
# Define the desired order of species
list_of_spec <- c("Acer p.", "Aesculus h.", "Betula p.", "Carpinus b.", "Fagus s.", "Fraxinus e.",
                  "Prunus a.", "Quercus p.", "Sorbus a.", "Tilia p.", "Grass")

###remove all stored object in the environment
#rm(list=ls())



png(paste0(data_path,"F1_per_class_final.png"), width = 1400, height = 960, res = 300)



#for vline settings
species_order <- with(PerclasF1, unique(Species[order(Plot_type)]))
breaks <- which(diff(as.numeric(factor(species_order, levels = list_of_spec))) != 1) + 0.5

#color keys for catagories
colors <- c("1" = "#3F8FCF", "2" = "#3F8FCF", "4" = "#D1E9F7")
# Get the number of unique types
num_types <- length(unique(PerclasF1$Plot_type))
ggplot(PerclasF1, aes(x = factor(Species, levels = list_of_spec), y = F1)) +
  geom_boxplot(aes(fill = factor(Plot_type, levels = sort(unique(Plot_type)))), color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +
  geom_vline(xintercept = breaks, linetype = "dashed", color = "#1F618D", size = 0.1, alpha = 0.5) +
  labs(title = "Per-tile F1 scores by tree species", y = "F1", fill = "Species\nCount") +
  scale_fill_manual(values = colors)+
  theme_classic() +
  theme(
    axis.title = element_text(size = 8, face = "bold"),
    axis.text = element_text(size = 8),
    plot.title = element_text(size = 8, face = "bold"),
    legend.position = "right",
    legend.title = element_text(size = 8, face = "bold"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9),
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
  )

dev.off()




