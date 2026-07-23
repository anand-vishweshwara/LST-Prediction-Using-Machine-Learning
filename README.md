# Land Surface Temperature Prediction using Machine Learning

## Overview

This project predicts **Land Surface Temperature (LST)** using Machine Learning techniques based on environmental and geographical features. Three regression models—**Random Forest**, **AdaBoost**, and **XGBoost**—are trained and compared to evaluate their prediction performance.

---

## Features

- Data preprocessing and missing value handling
- Feature selection
- Random Forest Regression
- AdaBoost Regression
- XGBoost Regression
- Feature importance analysis
- Model comparison
- Performance evaluation using multiple metrics
- Data visualization using Matplotlib

---

## Dataset

The dataset contains the following input features:

- NDVI (Normalized Difference Vegetation Index)
- Humidity (%)
- Wind Speed (m/s)
- Built-up Area (%)
- Longitude
- Latitude

Target Variable:

- Land Surface Temperature (°C)

---

## Machine Learning Workflow

1. Load the dataset
2. Handle missing values
3. Select input features
4. Split the dataset into training and testing sets
5. Train:
   - Random Forest Regressor
   - AdaBoost Regressor
   - XGBoost Regressor
6. Predict Land Surface Temperature
7. Evaluate model performance
8. Visualize feature importance and prediction results

---

## Performance Metrics

The following metrics are used to compare the models:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Results

### Feature Importance

![Feature Importance](feature_importance.png)

### Random Forest Prediction

![Random Forest](random_forest_prediction.png)

### AdaBoost Prediction

![AdaBoost](adaboost_prediction.png)

### XGBoost Prediction

![XGBoost](xgboost_prediction.png)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost

---

## How to Run

1. Install the required libraries.
2. Place the dataset in the project directory.
3. Run:

```bash
python lst_prediction.py
```

4. View the evaluation metrics and plots.

---

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Interactive Streamlit web application
- Model deployment

---

## Author

**Anand Vishweshwara**

Electronics and Communication Engineering (ECE)

Vasavi College of Engineering

Hyderabad, India
