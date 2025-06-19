# Breast Cancer Classification Using Logistic Regression

This repository contains a machine learning project that uses **logistic regression** to classify breast tumors as malignant or benign based on the Breast Cancer Wisconsin Diagnostic dataset from `sklearn.datasets`.



## Dataset

The Breast Cancer Wisconsin dataset contains 569 samples with 30 numerical features computed from digitized images of fine needle aspirate (FNA) of breast masses. The target is binary:

- `0` = Malignant  
- `1` = Benign

The features describe characteristics of cell nuclei such as radius, texture, perimeter, area, smoothness, concavity, and symmetry.


## Project Overview

- Split data into training (80%) and testing (20%) sets  
- Standardized features using `StandardScaler`  
- Trained a logistic regression model to predict tumor type  
- Evaluated model using accuracy, confusion matrix, classification report, and ROC AUC score  
- Visualized results with confusion matrix heatmap and ROC curve  

## Performance Summary

- **Accuracy:** 97.4%  
- **ROC AUC:** 0.997  
- **Precision & Recall:** Above 0.96 for both classes  
- Only 3 misclassifications on the test set

## Dependencies    

All dependencies can be installed using:  
```bash
pip install -r requirements.txt
```

The notebook also relies on helper functions defined in `functions.py`, which are used to sabstract some code. Make sure `functions.py` is in the same directory as the notebook for it to work properly.

## Usage

1. **Run the analysis**: Open and execute the `Breast Cancer.ipynb` notebook to walk through the full analysis.
3. **Function Support**: The notebook relies on helper functions in `functions.py` to abstract some operations.
4. **View Results**: After running the notebook, explore the visual outputs, summary statistics, and correlations directly within the file. 
Since github does not render the graphs created with plotly please check out `Breast Cancer.html` file.


