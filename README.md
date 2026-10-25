# House Price Prediction using Linear Regression

A beginner-level machine learning project that predicts house prices from area, number of bedrooms, number of bathrooms and age of the house using Linear Regression (scikit-learn).

## Dataset
The dataset is synthetic. If `house_data.csv` is not found, the script generates 1000 rows using a fixed random seed, with a known pricing formula plus random noise. About 15 values in `age_years` are deliberately set to missing to demonstrate data cleaning. Because the data is synthetic, the model scores are high by design and do not represent real-world performance.

## Features
- `area_sqft`, `bedrooms`, `bathrooms`, `age_years`

Target: `price`

## What the script does
1. Loads or generates the dataset
2. Cleans data (missing values filled with the training-set median, duplicates removed)
3. Explores data (shape, summary statistics, correlation with price)
4. Splits data into 80% training and 20% testing
5. Trains a Linear Regression model
6. Evaluates using RMSE and R2 score
7. Prints model coefficients and intercept
8. Saves the model as `house_model.pkl` and predicts two example houses

## How to run
pip install -r requirements.txt
python house_price_prediction.py

## Results
Coefficients (price change per 1 unit increase):
  area_sqft : 149.66
  bedrooms  : 10,139.24
  bathrooms : 9,569.03
  age_years : -841.12
  Intercept : -4,009.80

## Tech stack
Python, pandas, NumPy, scikit-learn, joblib

## Possible improvements
- Use a real-world dataset
- Try Ridge/Lasso regression and cross-validation
- Build a simple web app for predictions
