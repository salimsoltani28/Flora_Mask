# Automated Mask Generation for Vegetation Monitoring Using Citizen Science Data

Welcome to the repository for our study on automating mask generation for vegetation monitoring! This project explores the integration of **Segment Anything Model (SAM)** and **Grad-CAM** to create per-pixel segmentation masks from crowd-sourced plant photographs, enabling the training of segmentation models with minimal manual effort. Our approach leverages citizen science platforms like iNaturalist and Pl@ntNet, along with high-resolution UAV imagery, to advance scalable and cost-effective biodiversity monitoring.

---

## **Overview**

Manual annotation of training data for remote sensing and vegetation mapping is labor-intensive and often a bottleneck for machine learning applications in ecology. In this study, we propose an automated workflow that:

- Utilizes **SAM** for segmentation tasks and **Grad-CAM** for feature attribution to generate training masks.
- Incorporates citizen science photographs as a data source for training segmentation models.
- Demonstrates transferability and robustness of models for mapping diverse vegetation types.

By bridging citizen science data with UAV-based remote sensing, our workflow offers a scalable alternative to traditional manual annotation, significantly reducing the time and effort required for large-scale vegetation monitoring.

---

## **Key Features**

- **Automated Labeling**: Combines SAM and Grad-CAM to automate the creation of per-pixel segmentation masks from crowd-sourced images.
- **Direct Training from Citizen Science Data**: Bypasses the need for manually labeled UAV data by directly utilizing annotated plant photographs.
- **Scalable Workflow**: Designed for UAV orthoimagery, allowing application across diverse landscapes and vegetation compositions.
- **Performance Validation**: Evaluated on UAV orthoimages containing ten temperate deciduous tree species, achieving varying F1 scores across species.

---

## **Why This Matters**

This study demonstrates the feasibility of automating segmentation mask generation for ecological applications. By integrating citizen science data with state-of-the-art AI techniques, we provide a practical solution for:

- **Biodiversity Monitoring**: Tracking plant species distributions and changes over time.
- **Conservation Planning**: Enabling large-scale mapping to inform ecological management.
- **Cost Efficiency**: Reducing dependency on labor-intensive manual annotation processes.

---

## **Repository Contents**

1. **Code**: Scripts for training models, applying SAM and Grad-CAM, and generating segmentation masks.
2. **Data Examples**: Example UAV orthoimages and crowd-sourced plant photographs used in this study.
3. **Documentation**: Detailed instructions for setting up, running the workflow, and reproducing the results.
4. **Results**: Performance metrics (F1 scores, precision, recall) for the trained models applied to UAV imagery.

---

## **Getting Started**

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/vegetation-monitoring-masks.git
   cd vegetation-monitoring-masks
