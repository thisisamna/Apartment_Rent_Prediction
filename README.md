# Apartment Rent Prediction Project

## Overview

This project focuses on predicting apartment rent using machine learning models. The project tackles two main tasks:

1. **Regression**: Predicting the exact rental price of an apartment.
2. **Classification**: Predicting the price class (e.g., low, medium, high) of an apartment.

## Project Structure

The project is organized as follows:

```
Apartment-Rent-Prediction/
│
├── Classification/
│   ├── Classification_Dataset.csv
│   ├── Classification_Training.py
│   ├── classification.pkl
│   └── Classification_Test.py
│
├── Regression/
│   ├── Regression_Dataset.csv
│   ├── Regression_Training.py
│   ├── resgression.pkl
│   └── Regression_Test.py
│
├── requirements.txt
└── README.md
```

## Dependencies

The project relies on several key Python libraries. To install them, run:

```bash
pip install -r requirements.txt
```

**Key dependencies include:**

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- xgboost

## Files

Both regression and classification directories contains the following files:

- `_Dataset.csv`: Contains the original, unprocessed data
- `_Training.py`: Script that preprocesses the data, performs feature selection, trains different models and saves results to \_.pkl file.
- `_pkl`: Contains the trained models as well as encoders and any learned values to be used in preprocessing.
- `_test.py`: Test/deployment script that loads the pickle file, preprocesses the data, makes predictions and evaluates the model.

## Results

Model performance metrics are printed upon running any of the scripts (training or testing).

---

Thank you for checking out the Apartment Rent Prediction Project!
