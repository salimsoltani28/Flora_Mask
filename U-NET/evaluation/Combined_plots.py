# Required Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import numpy as np

def combined_single_plot(f1_data_path, cm_data_path, output_path):
    # Step 1: Load and Clean Data for F1 Score Plot
    data = pd.read_csv(f1_data_path)

    # Filter rows where 'Plot_type' ends with "1"
    data = data[data['Plot_type'].str.endswith("1")]

    # Define the correct species order
    list_of_spec = [
        "Acer.p", "Aesculus.h", "Betula.p", "Carpinus.b", "Fagus.s", 
        "Fraxinus.e", "Prunus.a", "Quercus.p", "Sorbus.a", "Tilia.p", "Grass"
    ]

    # Update 'Species' to use the correct order and sort
    data['Species'] = pd.Categorical(data['Species'], categories=list_of_spec, ordered=True)
    data.sort_values('Species', inplace=True)

    # Step 2: Load and Process Data for Confusion Matrix Plot
    confusion_matrix_data = pd.read_csv(cm_data_path, index_col=0)
    normalized_cm = confusion_matrix_data.div(confusion_matrix_data.sum(axis=1), axis=0) * 100

    # Step 3: Create Combined Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18), sharex=True, gridspec_kw={
        'height_ratios': [1, 1], 'hspace': 0
    })

    # F1 Score Plot
    sns.boxplot(data=data, x='Species', y='F1', color="#3F8FCF", linewidth=0.5, fliersize=0, showfliers=False, ax=ax1)
    ax1.set_title("")  # Remove title
    ax1.set_xlabel("")  # Remove x-axis label
    ax1.set_ylabel("F1 Score", fontsize=16, fontweight='bold')  # Y-axis font size
    ax1.tick_params(axis='both', which='major', labelsize=16)  # X and Y ticks font size
    ax1.grid(False)  # Remove horizontal grid lines

    # Add dashed vertical lines as column separators in the first plot
    for i in range(len(list_of_spec) - 1):
        ax1.axvline(i + 0.5, color='gray', linestyle='--', linewidth=0.8)

    # Add a box frame around the top figure
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    # Confusion Matrix Plot
    im = ax2.imshow(normalized_cm, cmap=plt.cm.Blues, aspect='auto')

    list_of_spec = [ # Update list of species to include Background 
        "Acer.p", "Aesculus.h", "Betula.p", "Carpinus.b", "Fagus.s",
        "Fraxinus.e", "Prunus.a", "Quercus.p", "Sorbus.a", "Tilia.p", "Background"]
    
    # Add Confusion Matrix Labels
    for (i, j), val in np.ndenumerate(normalized_cm):
        ax2.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=16, color='black' if val < 50 else 'white')

    # Modify x-axis and y-axis labels
    ax2.set_xlabel("Tree Species (predicted)", fontsize=16, fontweight='bold')  # X-axis label font size
    ax2.set_ylabel("Tree Species", fontsize=16, fontweight='bold')  # Y-axis label font size
    ax2.tick_params(axis='both', which='major', labelsize=16)  # X and Y ticks font size
    ax2.set_xticks(np.arange(len(list_of_spec)))
    ax2.set_xticklabels([f"$\it{{{s}}}$" for s in list_of_spec], rotation=45, ha='right', fontsize=20, fontweight='bold')
    ax2.set_yticks(np.arange(len(list_of_spec)))
    ax2.set_yticklabels([f"$\it{{{s}}}$" for s in list_of_spec], fontsize=20, fontweight='bold')

    # Add a box frame around the confusion matrix
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color("black")

    # Add dashed vertical and horizontal lines to separate columns and rows
    for i in range(len(list_of_spec) - 1):
        ax2.axvline(i + 0.5, color='gray', linestyle='--', linewidth=0.8)
        ax2.axhline(i + 0.5, color='gray', linestyle='--', linewidth=0.8)

    # Add Colorbar to Margin
    #cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])
    #fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    # Adjust Colorbar to avoid overlap and fit within the adjusted layout
    cbar_ax = fig.add_axes([0.9, 0.14, 0.02, 0.35])
    #fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    # Add Colorbar to Margin
    #cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # Adjust position as needed
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')

# Add Title to Colorbar
    cbar.set_label("Confusion Matrix (% per Class)", fontsize=16, fontweight='bold', labelpad=20)

    # Adjust Layout: Overlap x-axis of top plot with bottom border
    #plt.subplots_adjust(top=0.97, bottom=0.1, left=0.1, right=0.88, hspace=0)
        # Adjust Layout: Shrink content from all sides
    plt.subplots_adjust(top=0.96, bottom=0.14, left=0.15, right=0.87, hspace=0.2)

    


    # Save and Close Plot
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Updated combined plot saved as '{output_path}'")




# File paths
f1_data_path = 'Z:/projects/bigplantsens/2_UNET_on_Flora_Mask/01_MyDiv/Pred_output/Ortho_July_/EXP3_over0.2under20_stem_30percent_cls11_Ortho_backRplcd_ZoomEXCLD_img512_80/Majority_vote/meanTreespecF1_0.25_F1gr0.5_1_per_tile_forallplots.csv'
cm_data_path = 'Z:/projects/bigplantsens/2_UNET_on_Flora_Mask/01_MyDiv/Pred_output/Ortho_July_/EXP3_over0.2under20_stem_30percent_cls11_Ortho_backRplcd_ZoomEXCLD_img512_80/Majority_vote/Total_confusion_matrix.csv'
output_path = 'Z:/projects/bigplantsens/2_UNET_on_Flora_Mask/01_MyDiv/Pred_output/Ortho_July_/EXP3_over0.2under20_stem_30percent_cls11_Ortho_backRplcd_ZoomEXCLD_img512_80/Majority_vote/Combined_F1_and_CM_Updated_Plot_backg.png'

# Generate combined plot with updated settings
combined_single_plot(f1_data_path, cm_data_path, output_path)
