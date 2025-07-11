# Wound Border Characterization
**Quantitative 3D Depth-Based Analysis and Machine Learning-Based Classification of Chronic Wound Borders**


## Project Description
This project implements a fully automated pipeline for the objective assessment and classification of chronic wound borders using 3D depth imaging and machine learning. Chronic wounds—such as diabetic foot ulcers, pressure ulcers, and venous leg ulcers—pose a serious challenge in clinical care due to their slow healing and the subjective nature of visual assessments.

To overcome the limitations of manual evaluation (which often suffers from inter-observer variability and inconsistency), this project builds on recent methods using depth maps and quantitative geometry analysis to provide a reproducible and interpretable machine learning framework. It offers both unsupervised discovery of wound edge types and supervised classification using data-driven features derived from standardized wound border representations.

##  Project Overview
### Objectives 
To design, implement, and evaluate an end-to-end computational pipeline that:
- Converts irregular wound shapes into a standardized rectified form
- Extracts meaningful geometric and statistical features from 3D profiles
- Identifies natural groupings of wound types without prior labels
- Trains a machine learning model to classify new wounds based on discovered types
- Analyzes which geometric properties most influence classification

### Dataset Overview
**Note**: The dataset used in this project is not publicly available due to clinical data privacy restrictions.
The project uses wound cases that include:
- RGB wound images captured in clinical settings
- Corresponding **3D depth maps**
- **Binary segmentation masks** for both the wound area and the surrounding body region

These inputs allow the pipeline to perform accurate depth correction, border rectification, and quantitative feature analysis.

---
## Pipeline Summary

1. **Depth Map Filtering and Correction**  
   Cleans raw depth data and removes body curvature using 3D surface fitting.

2. **Wound Border Rectification**  
   Reshapes the irregular wound border into a standardized rectangular format.

3. **Mean Depth Profile Generation**  
   Creates a 1D smoothed cross-section of the border for feature extraction.

4. **Feature Extraction**  
   Extracts 28 features: statistical, curve-fitting, and spectral descriptors.

5. **Unsupervised Clustering**  
   Uses PaCMAP + HDBSCAN to discover natural wound border types.

6. **Supervised Classification**  
   Trains a Random Forest to classify new wounds and determine key features.

---

## Key Features

- Fully automated and scriptable pipeline
- Shape-independent wound border representation
- High-performance Random Forest classification (82.6% accuracy)
- Feature importance ranking for interpretability
- Clinical use potential for standardizing wound assessment

## 📁 Folder Structure

```bash
Wound-Border-Characterization/
│
├── data/                      # Raw input data (not publicly shared)
│   ├── body_masks/            # Binary masks for body region
│   ├── depth_maps/            # 16-bit depth maps of wounds
│   ├── images/                # Original RGB wound images
│   ├── marker_masks/          # Optional additional masks (if used)
│   └── wound_masks/           # Binary masks outlining wound regions
│
├── metadata/                   # Metadata for the dataset
│   ├── image_index.csv
│   └── image_index_filtered.csv
│
├── notebooks/                  # Interactive Jupyter notebooks for each step
│   ├── 01_pipeline_walkthrough.ipynb       # Overview of the full pipeline
│   ├── 02_unsupervised_clustering.ipynb    # Clustering using PaCMAP + HDBSCAN
│   └── 03_supervised_classification.ipynb  # Random Forest classifier training
│
├── outputs/                    # Generated files from pipeline execution (Publically not available. This will update upon run each pipelines)
│   ├── cluster_profiles.csv             # Sample IDs per cluster
│   ├── cluster_summary_stats.csv        # Mean/SD of features per cluster
│   ├── comprehensive_features.csv       # All extracted features
│   ├── features_with_labels.csv         # Features with cluster labels
│   └── image_cluster_map.csv            # Mapping of images to clusters
│
├── src/                        # Python source code (modular implementation)
│   ├── __init__.py
│   ├── data_loader.py          # I/O functions for loading images and masks
│   ├── feature_extraction.py   # Main functions for 1D profile generation + feature extraction
│   ├── filter_data.py          # Preprocessing: image filtering, wound selection
│   ├── plotting.py             # Visualization utilities (profiles, clusters, confusion matrix)
│   ├── preprocessing.py        # Z-score filtering and depth correction
│   └── utils.py                # Helper functions (e.g., sigmoid fitting)
│
├── run_pipeline.py             # Script to run for feature extraction pipeline (CLI or headless mode)
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package metadata 
└── README.md                   # Project overview and usage guide


## 🚀 Getting Started

### Prerequisites
To run this project, you will need:
* Python 3.8 or higher
* `pip` (Python package installer)
* Git

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MPSVIRAJ/Wound-Border-Characterization.git]
    cd Wound-Border-Characterization
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
