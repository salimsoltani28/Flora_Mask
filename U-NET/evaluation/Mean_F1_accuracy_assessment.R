#tempdir(path="/scratch1/ssoltani/workshop/00 Code/07 Accuracy Assessment/01_yDiv_tree_species/")
.libPaths("/home/ms2487/miniconda3/envs/r_env/lib/R/library")
Sys.setenv(PROJ_LIB = "/home/ms2487/miniconda3/envs/r_env/share/proj")

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





# Extract the directory path from the command line arguments
path_pred <- "Z:/projects/bigplantsens/2_UNET_on_Flora_Mask/01_MyDiv/Pred_output/Ortho_July_/EXP3_over0.2under20_stem_30percent_cls11_Ortho_backRplcd_ZoomEXCLD_img512_80/Majority_vote/"


Acc_val <- paste0(path_pred, "meanTreespecF1_0.25_F1gr0.5_1_per_tile_forallplots.csv") %>% read.csv()
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
  mutate(Plot_type=replace_pattern(Plot_type))
# Generate and save the plot for the complete data

# # Replace the values
# PerclasF1$Species <- gsub("\\.", " ", PerclasF1$Species)
# PerclasF1$Species <- gsub("([^\\.])$", "\\1.", PerclasF1$Species)
# PerclasF1$Species <- ifelse(PerclasF1$Species == "Grass.", "Grass", PerclasF1$Species)



# Alternatively, in a more compact form:
PerclasF1$Species <- ifelse(gsub("([^\\.])$", "\\1.", gsub("\\.", " ", PerclasF1$Species)) == "Grass.", "Grass", gsub("([^\\.])$", "\\1.", gsub("\\.", " ", PerclasF1$Species)))

Acc_val <- PerclasF1 %>% filter(Plot_type==1)
# Compute mean F1 score for each list of F1 scores
No_class_F1_high <- Acc_val %>% group_by(Species) %>% 
  summarize(meanf1=mean(F1)) %>% 
  filter(Species!="Grass" & meanf1>0.5) %>% count()

# Compute mean F1 score for each list of F1 scores
Treeclass_mean <- Acc_val %>% group_by(Species) %>% 
  summarize(meanf1=mean(F1)) #%>% 
  #filter(Species!="Grass" ) %>% pull(meanf1) %>% mean() %>% 
  #round(digits = 2)


# Compute mean F1 score for each list of F1 scores
No_class_F1_high <- Acc_val %>% group_by(Species) %>% 
  summarize(meanf1=mean(F1)) %>% 
  filter(Species!="Grass" & meanf1>0.5) %>% count()

# Compute mean F1 score for each list of F1 scores
Treeclass_mean <- Acc_val %>% group_by(Species) %>% 
  summarize(meanf1=mean(F1)) #%>% 
#filter(Species!="Grass" ) %>% pull(meanf1) %>% mean() %>% 
#round(digits = 2)



#png(paste0(pathcsv,"Box_plot_perPlot_F1.png"), width = 600, height = 600,res =600)
png(paste0(path_pred,"completeData_box_perTile_F1.png"), width = 1200, height = 1200, res = 300)



# Assume that your raster values are integers from 1 to 11
# (Adjust if your raster has a different range of values)
#color_palette <- setNames(colors, 1:11)

# Define the desired order of species
list_of_spec <- c("Acer p.", "Aesculus h.", "Betula p.", "Carpinus b.", "Fagus s.", "Fraxinus e.",
                  "Prunus a.", "Quercus p.", "Sorbus a.", "Tilia p.", "Grass")

# Define colors for each species
species_colors <- c(
  "Acer p." = "#8dd3c7",
  "Aesculus h." = "#ffffb3",
  "Betula p." = "#bebada",
  "Carpinus b." = "#fb8072",
  "Fagus s." = "#80b1d3",
  "Fraxinus e." = "#fdb462",
  "Prunus a." = "#b3de69",
  "Quercus p." = "#fccde5",
  "Sorbus a." = "#d9d9d9",
  "Tilia p." = "#bc80bd",
  "Grass" = "#ccebc5"
)

ggplot(PerclasF1, aes(x = factor(Species, levels = list_of_spec), y = F1, fill = Species)) +
  geom_boxplot( color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +  # Adjust the size parameter
  scale_fill_manual(values = species_colors) +
  labs(title = "Per-tile F1 scores by tree species", y = "F1", x = FALSE) +
  theme_classic() +
  theme(axis.title = element_text(size = 8, face = "bold"),
        axis.text = element_text(size = 8),
        plot.title = element_text(size = 8, face = "bold"),
        legend.position = "none",
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        axis.title.x = element_blank())




dev.off()

###remove all stored object in the environment
#rm(list=ls())



png(paste0(path_pred,"completeData_Species_composition_F1_by_class.png"), width = 1800, height = 960, res = 300)



#for vline settings
species_order <- with(PerclasF1, unique(Species[order(Plot_type)]))
breaks <- which(diff(as.numeric(factor(species_order, levels = list_of_spec))) != 1) + 0.5

#color keys for catagories
colors <- c("1" = "#002855", "2" = "#3F8FCF", "4" = "#D1E9F7")
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






##########################################################################Comparing between the orthos
#exclude plots that dont overlap
No_overlapplots <- AOI@data %>% 
  filter(Ortho2_inc==0) %>% pull(Plot_No)

PerclasF1_filter <- PerclasF1 %>% 
  filter(!Plot_type %in% No_overlapplots)



# Generate and save the plot for the filtered data


#complete data
PerclasF1_filter <- PerclasF1_filter %>% 
  mutate(Plot_type=replace_pattern(Plot_type))
# Generate and save the plot for the complete data






#png(paste0(pathcsv,"Box_plot_perPlot_F1.png"), width = 600, height = 600,res =600)
png(paste0(path_pred,"Comparision_box_perTile_F1.png"), width = 1200, height = 1200, res = 300)



# Assume that your raster values are integers from 1 to 11
# (Adjust if your raster has a different range of values)
#color_palette <- setNames(colors, 1:11)





ggplot(PerclasF1_filter, aes(x = factor(Species, levels = list_of_spec), y = F1, fill = Species)) +
  geom_boxplot( color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +  # Adjust the size parameter
  scale_fill_manual(values = species_colors) +
  labs(title = "Per-tile F1 scores by tree species", y = "F1", x = FALSE) +
  theme_classic() +
  theme(axis.title = element_text(size = 8, face = "bold"),
        axis.text = element_text(size = 8),
        plot.title = element_text(size = 8, face = "bold"),
        legend.position = "none",
        panel.border = element_rect(color = "black", fill = NA, linewidth = 0.9),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
        axis.title.x = element_blank())




dev.off()

###remove all stored object in the environment
#rm(list=ls())



png(paste0(path_pred,"Comparision_Species_composition_F1_by_class.png"), width = 1800, height = 960, res = 300)



#forplot vline
species_order <- with(PerclasF1_filter, unique(Species[order(Plot_type)]))
breaks <- which(diff(as.numeric(factor(species_order, levels = list_of_spec))) != 1) + 0.5
#catagories color
colors <- c("1" = "#002855", "2" = "#3F8FCF", "4" = "#D1E9F7")
# Get the number of unique types
num_types <- length(unique(PerclasF1_filter$Plot_type))
ggplot(PerclasF1_filter, aes(x = factor(Species, levels = list_of_spec), y = F1)) +
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






#####################


# #add the plot type
# # Function to replace patterns with values
# replace_pattern <- function(data) {
#   # Extract numeric values after the letters
#   values <- gsub("^.*[A-Za-z]+([0-9]+)$", "\\1", data)
#   
#   # Convert the extracted values to numeric
#   values <- as.numeric(values)
#   
#   return(values)
# }
# 
# #complete data
# PerclasF1 <- PerclasF1 %>% 
#   mutate(Plot_type=replace_pattern(Plot_type))
# # Generate and save the plot for the complete data
# 
# 
# 
# 
# 
# 
# #png(paste0(pathcsv,"Box_plot_perPlot_F1.png"), width = 600, height = 600,res =600)
# png(paste0(path_pred,"completeData_box_perTile_F1.png"), width = 1200, height = 1200, res = 300)
# 
# 
# 
# # Assume that your raster values are integers from 1 to 11
# # (Adjust if your raster has a different range of values)
# #color_palette <- setNames(colors, 1:11)
# 
# # Define the desired order of species
# list_of_spec <- c("Acer.p", "Aesculus.h", "Betula.p", "Carpinus.b", "Fagus.s", "Fraxinus.e",
#                   "Prunus.a", "Quercus.p", "Sorbus.a", "Tilia.p", "Grass")
# 
# # Define colors for each species
# species_colors <- c(
#   "Acer.p" = "#8dd3c7",
#   "Aesculus.h" = "#ffffb3",
#   "Betula.p" = "#bebada",
#   "Carpinus.b" = "#fb8072",
#   "Fagus.s" = "#80b1d3",
#   "Fraxinus.e" = "#fdb462",
#   "Prunus.a" = "#b3de69",
#   "Quercus.p" = "#fccde5",
#   "Sorbus.a" = "#d9d9d9",
#   "Tilia.p" = "#bc80bd",
#   "Grass" = "#ccebc5"
# )
# 
# ggplot(PerclasF1, aes(x = factor(Species, levels = list_of_spec), y = F1, fill = Species)) +
#   geom_boxplot( color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +  # Adjust the size parameter
#   scale_fill_manual(values = species_colors) +
#   labs(title = "Per-tile F1 scores by tree species", y = "F1", x = FALSE) +
#   theme_classic() +
#   theme(axis.title = element_text(size = 8, face = "bold"),
#         axis.text = element_text(size = 8),
#         plot.title = element_text(size = 8, face = "bold"),
#         legend.position = "none",
#         panel.border = element_rect(color = "black", fill = NA, size = 0.9),
#         axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
#         axis.title.x = element_blank())
# 
# 
# 
# 
# dev.off()
# 
# ###remove all stored object in the environment
# #rm(list=ls())
# 
# 
# 
# png(paste0(path_pred,"completeData_Species_composition_F1_by_class.png"), width = 1800, height = 960, res = 300)
# 
# 
# 
# 
# 
# # Get the number of unique types
# num_types <- length(unique(PerclasF1$Plot_type))
# ggplot(PerclasF1, aes(x = factor(Species, levels = list_of_spec), y = F1)) +
#   geom_boxplot(aes(fill = factor(Plot_type, levels = sort(unique(Plot_type)))), color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +
#   geom_vline(xintercept = seq(1.5, num_types - 0.5), linetype = "dashed", color = "#1F618D", size = 0.1, alpha = 0.5) +
#   labs(title = "Per-tile F1 scores by tree species", y = "F1", fill = "Species\nCount") +
#   theme_classic() +
#   theme(
#     axis.title = element_text(size = 8, face = "bold"),
#     axis.text = element_text(size = 8),
#     plot.title = element_text(size = 8, face = "bold"),
#     legend.position = "right",
#     legend.title = element_text(size = 8, face = "bold"),
#     panel.border = element_rect(color = "black", fill = NA, size = 0.9),
#     axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
#   )
# 
# dev.off()
# 
# 
# 
# 
# 
# 
# ##########################################################################Comparing between the orthos
# #exclude plots that dont overlap
# No_overlapplots <- AOI@data %>% 
#   filter(Ortho2_inc==0) %>% pull(Plot_No)
# 
# PerclasF1_filter <- Acc_val %>% 
#   filter(!Plot_type %in% No_overlapplots)
# 
# 
# 
# # Generate and save the plot for the filtered data
# 
# 
# #complete data
# PerclasF1_filter <- PerclasF1_filter %>% 
#   mutate(Plot_type=replace_pattern(Plot_type))
# # Generate and save the plot for the complete data
# 
# 
# 
# 
# 
# 
# #png(paste0(pathcsv,"Box_plot_perPlot_F1.png"), width = 600, height = 600,res =600)
# png(paste0(path_pred,"Comparision_box_perTile_F1.png"), width = 1200, height = 1200, res = 300)
# 
# 
# 
# # Assume that your raster values are integers from 1 to 11
# # (Adjust if your raster has a different range of values)
# #color_palette <- setNames(colors, 1:11)
# 
# # Define the desired order of species
# list_of_spec <- c("Acer.p", "Aesculus.h", "Betula.p", "Carpinus.b", "Fagus.s", "Fraxinus.e",
#                   "Prunus.a", "Quercus.p", "Sorbus.a", "Tilia.p", "Grass")
# 
# # Define colors for each species
# species_colors <- c(
#   "Acer.p" = "#8dd3c7",
#   "Aesculus.h" = "#ffffb3",
#   "Betula.p" = "#bebada",
#   "Carpinus.b" = "#fb8072",
#   "Fagus.s" = "#80b1d3",
#   "Fraxinus.e" = "#fdb462",
#   "Prunus.a" = "#b3de69",
#   "Quercus.p" = "#fccde5",
#   "Sorbus.a" = "#d9d9d9",
#   "Tilia.p" = "#bc80bd",
#   "Grass" = "#ccebc5")
# 
# ggplot(PerclasF1_filter, aes(x = factor(Species, levels = list_of_spec), y = F1, fill = Species)) +
#   geom_boxplot( color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +  # Adjust the size parameter
#   scale_fill_manual(values = species_colors) +
#   labs(title = "Per-tile F1 scores by tree species", y = "F1", x = FALSE) +
#   theme_classic() +
#   theme(axis.title = element_text(size = 8, face = "bold"),
#         axis.text = element_text(size = 8),
#         plot.title = element_text(size = 8, face = "bold"),
#         legend.position = "none",
#         panel.border = element_rect(color = "black", fill = NA, size = 0.9),
#         axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
#         axis.title.x = element_blank())
# 
# 
# 
# 
# dev.off()
# 
# ###remove all stored object in the environment
# #rm(list=ls())
# 
# 
# 
# png(paste0(path_pred,"Comparision_Species_composition_F1_by_class.png"), width = 1800, height = 960, res = 300)
# 
# 
# 
# 
# 
# # Get the number of unique types
# num_types <- length(unique(PerclasF1_filter$Plot_type))
# ggplot(PerclasF1_filter, aes(x = factor(Species, levels = list_of_spec), y = F1)) +
#   geom_boxplot(aes(fill = factor(Plot_type, levels = sort(unique(Plot_type)))), color = "#1F618D", alpha = 0.7, outlier.shape = NA, size = 0.2) +
#   geom_vline(xintercept = seq(1.5, num_types - 0.5), linetype = "dashed", color = "#1F618D", size = 0.1, alpha = 0.5) +
#   labs(title = "Per-tile F1 scores by tree species", y = "F1", fill = "Species\nCount") +
#   theme_classic() +
#   theme(
#     axis.title = element_text(size = 8, face = "bold"),
#     axis.text = element_text(size = 8),
#     plot.title = element_text(size = 8, face = "bold"),
#     legend.position = "right",
#     legend.title = element_text(size = 8, face = "bold"),
#     panel.border = element_rect(color = "black", fill = NA, size = 0.9),
#     axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)
#   )
# 
# dev.off()
# 
