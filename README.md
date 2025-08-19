# Wound Border Characterization
**Quantitative 3D Depth-Based Analysis and Machine Learning-Based Classification of Chronic Wound Borders**

Wound Border Characterization is a fully automated pipeline developed under a novel method to classify wound border types using a large set of quantitative features extracted from wound borders in 3D depth maps.
The pipeline includes three core functionalities:
- Quantitative Feature Extraction
- Unsupervised Edge Type Discovery
- Automated Classification
This project can be used in two ways: as a complete pipeline to train a new classification model on a custom dataset, or as a standalone tool to classify wounds using the included pre-trained model.

## 🧬 Pipeline Workflow

The project is structured as an end-to-end pipeline that processes raw 3D image data to produce a final classification. The diagram below illustrates the major stages of this process.

![Wound Border Characterization Pipeline](docs/source//process.png)
*Figure 1: The complete pipeline, from initial data preprocessing and feature extraction to unsupervised clustering for label discovery and final supervised classification for prediction.*

### A Note on Classification Labels
Please note that the classification labels used to train the model (e.g., "Shallow Bed, Gentle Slope") are not clinically verified. They were generated in a data-driven manner from the unsupervised clustering step based on statistical similarities in the wound geometry.

The primary goal of this project is to provide a robust framework for this type of analysis. Users can easily adapt the pipeline to use their own clinically verified labels by modifying the `DescriptiveLabels` section in the `config.json` file.


### Key features
* **Fully Documented:** The entire codebase is commented and includes detailed docstrings for all modules and functions.
* **End-to-End Pipeline:** Includes modules for preprocessing, quantitative feature extraction, clustering, classification, and visualization.
* **Thoroughly Tested:** A comprehensive suite of unit tests ensures code reliability, achieving over 90% coverage on core application logic.
* **CLI Controllable:** A robust command-line interface allows for easy execution of any individual pipeline stage (`extract`, `cluster`, `train`, `predict`) or the full sequence.


### 📊 The Dataset
The dataset for this study was provided by the IRCCS Sant' Orsola Malpighi University Hospital in Bologna. It originated from an initial pool of 7,329 mobile phone images and their corresponding 16-bit depth maps, which were captured during daily clinical routines. After a programmatic quality filtering process, a final set of 1,436 images was selected for the pipeline, from which **1,143** were successfully processed to generate the final feature set.

**Important:** Due to patient privacy and confidentiality, the clinical dataset is private and is **not included** in this repository.

To allow for a functional demonstration of the entire pipeline, a small, synthetic **sample dataset** is provided in the `/sample_data` directory. This sample data can be used to run all stages of the project to verify its functionality.

### **Using Your Own Data**
The pipeline is designed to work on custom datasets. To process your own images, you must provide the following four files for each sample, all sharing the same base filename (e.g., `image_001.png`):

* **RGB Image:** The original color photograph of the wound.
* **Wound Mask:** A binary (black and white) image where the wound area is white.
* **Body Mask:** A binary image where the entire anatomical part (e.g., the leg or arm) is white.
* **Depth Map:** A 16-bit single-channel image representing the 3D depth information.

These files must be placed in their corresponding subdirectories within the `/data` folder (`data/images`, `data/wound_masks`, etc.).

## 🚀 Getting Started
To get a local copy up and running, follow these simple steps.
### Prerequisites
This project can be set up using either a Conda distribution (like Anaconda or Miniforge) or a standard Python installation with Pip and Venv.
* Git
* Conda or Python 3.10+


### Installation
1.  **Clone the repository:**
    ```sh
    git clone https://github.com/MPSVIRAJ/Wound-Border-Characterization.git
    cd Wound-Border-Characterization
    ```
2.  **Create and activate a virtual environment (Optional):**
    
    
    On macOS/Linux:
    ```sh
    python3 -m venv wbc_venv
    source wbc_venv/bin/activate 
    ```

    On Windows:
    ```sh
    python -m venv wbc_venv
    .\wbc_venv\Scripts\activate       
    ```

3.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up the sample data**

    The repository includes a small, synthetic dataset to demonstrate the pipeline's functionality. Run the appropriate commands for your operating system to create a data directory and copy the sample files into it.
    

    On macOS / Linux:
    ```sh
    mkdir data
    cp -r sample_data/* data/
    ```

    On Windows (Command Prompt):
    ```sh
    mkdir data
    xcopy sample_data data /E /I
    ```

You are now ready to run the pipeline.

## ▶️ Usage
The entire pipeline is controlled via the main script **run_pipeline.py**. You must specify a **stage** to execute, which tells the script which part of the process to run.

**Important Note:** This repository includes the original, pre-computed feature file (outputs/1_comprehensive_features.csv)that was used to generate the results for the original study. This allows you to immediately evaluate the core machine learning components without needing the private image dataset.

The pipeline can be used in three primary ways depending on your goal.

### **Use Case 1: Classifying New Wounds with the Pre-trained Model**
If you have a new set of wound images (structured as described in "The Dataset" section) and want to classify them using the pre-trained model provided in this repository, use the **predict** command.

This command will first run feature extraction on your data and then use the existing trained model to predict the wound border type.

```sh
python run_pipeline.py predict
```
### **Use Case 2: Evaluating the Original Machine Learning Results**
If you wish to verify and reproduce the original clustering and model training process, you can run the following stages directly. These commands use the included feature file.
#### 1. Reproduce the clustering:
```sh
python run_pipeline.py cluster
```
#### 2. Reproduce the model training:
```sh
python run_pipeline.py train
```

### **Use Case 3: Training a New Model on a Custom Dataset**
If you have your own complete dataset and want to create a new model, you must run the pipeline stages step-by-step. Note: This process will overwrite the original feature file with features extracted from your images.

#### Step 1: Extract Features
Run the feature extraction on your entire dataset.
```sh
python run_pipeline.py extract
```
You can also visualize the process for a single image to ensure it's working as expected on your data:
```sh
python run_pipeline.py extract --image_id <your_image_id>
```
#### Step 2: Discover Clusters
Run the clustering stage to discover the natural groupings in your data.
```sh
python run_pipeline.py cluster
```
#### Step 3: Interpret Clusters and Update Config (Manual Step)
Stop here and inspect the output plots in the **outputs/** directory, especially samples_for_clusters.png. Create your own meaningful names for the new cluster labels that were discovered.

Then, open **config.json** and update the **DescriptiveLabels** section with your new labels.

#### Step 4: Train the New Model
Now, run the training stage. It will use your newly extracted features and your custom labels from the **config.json** to train a new classifier.
```sh
python run_pipeline.py train
```
After this step, your new custom-trained model is saved to the **outputs/** directory and is ready to be used for predictions.

## ✅ Tests
This project is committed to code quality and reliability, backed by a comprehensive suite of unit tests. The tests verify all core functionalities, edge cases, and error-handling routines.
### Running the Test Suite
To run all tests, execute the following command from the project root directory:
```sh
pytest -v
```
To generate a coverage report for the core application logic, run:
```sh
pytest --cov
```
### Testing Scope
**What is Tested:**

All modules containing core application logic are thoroughly tested. This includes:

* data loading and preprocessing routines
* Feature extraton process
* Clustering processes
* Classification tasks and other utility tasks

**What is Not Tested (By Design):**

Certain files are intentionally excluded from the unit test coverage metrics, following standard development practices:

* src/plotting.py: Visualization code is not unit tested, as it is complex to automate and provides low value. This aligns with the provided exam guidelines.
* src/logging_setup.py: This is a simple configuration script that is implicitly verified by the logging checks in all other tests.
* run_pipeline.py: As the main application entry point, this script's end-to-end functionality is verified by running the workflows described in the Usage section.

## 📖 Documentation
This project includes a complete documentation website that provides an in-depth overview of the project, its usage, and a detailed API reference for the entire codebase.

The documentation is automatically generated from the source code's docstrings using Sphinx and is hosted on GitHub Pages.

**[➡️ View the Full Documentation Here](https://mpsviraj.github.io/Wound-Border-Characterization/)**

## 📜 License
Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.
