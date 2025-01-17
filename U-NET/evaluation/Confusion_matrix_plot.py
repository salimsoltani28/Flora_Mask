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
        "Fraxinus.e", "Prunus.a", "Quercus.p", "Sorbus.a", "Tilia.p", "Background"
    ]

    # Update 'Species' to use the correct order and sort
    data['Species'] = pd.Categorical(data['Species'], categories=list_of_spec, ordered=True)
    data.sort_values('Species', inplace=True)

    # Step 2: Load and Process Data for Confusion Matrix Plot
    confusion_matrix_data = pd.read_csv(cm_data_path, index_col=0)
    normalized_cm = confusion_matrix_data.div(confusion_matrix_data.sum(axis=1), axis=0) * 100

    # Step 3: Create Combined Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True, gridspec_kw={
        'height_ratios': [1, 1], 'hspace': 0.05
    })

    # F1 Score Plot
    sns.boxplot(data=data, x='Species', y='F1', color="#3F8FCF", linewidth=0.5, fliersize=0, showfliers=False, ax=ax1)
    ax1.set_title("Per-tile F1 Scores by Tree Species", fontsize=14, fontweight='bold')
    ax1.set_ylabel("F1 Score", fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', bottom=False, labelbottom=False)
    ax1.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    # Confusion Matrix Plot (stretched to match above)
    im = ax2.imshow(normalized_cm, cmap=plt.cm.Blues, aspect='auto')

    # Add Confusion Matrix Labels
    for (i, j), val in np.ndenumerate(normalized_cm):
        ax2.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=8, color='black' if val < 50 else 'white')

    ax2.set_title("Confusion Matrix (Percentage per Class)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Tree Species", fontsize=12, fontweight='bold')
    ax2.set_ylabel("True Label", fontsize=12, fontweight='bold')
    ax2.set_xticks(np.arange(len(list_of_spec)))
    ax2.set_xticklabels(list_of_spec, rotation=45, ha='right')
    ax2.set_yticks(np.arange(len(list_of_spec)))
    ax2.set_yticklabels(list_of_spec)

    # Add Colorbar to Margin
    cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])  # Position of the colorbar
    fig.colorbar(im, cax=cbar_ax, orientation='vertical')

    # Save and Close Plot
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Combined single plot saved as '{output_path}'")

# File paths
#replace the backslash with forward slash
main_path = 'Z:/projects/bigplantsens/2_UNET_on_Flora_Mask/01_MyDiv/Pred_output/Ortho_July_/EXP3_over0.2under20_stem_30percent_cls11_Ortho_backRplcd_ZoomEXCLD_img512_80/Majority_vote/'
f1_data_path = main_path + 'meanTreespecF1_0.25_F1gr0.5_1_per_tile_forallplots.csv'
cm_data_path = main_path + 'Total_confusion_matrix.csv'
output_path = main_path + 'Combined_F1_and_CM_Aligned_Plot.png'

# Generate combined plot with aligned axes
combined_single_plot(f1_data_path, cm_data_path, output_path)
