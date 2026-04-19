# California Housing Price Regression

A machine learning project that predicts California housing prices using the sklearn California Housing dataset. Two regression models are implemented and compared: **Linear Regression** and **Decision Tree Regressor**.

## Overview

The notebook walks through a standard ML pipeline — loading data, preprocessing, training, and evaluating two regression models on the California Housing dataset.

## Dataset

**California Housing Dataset** (from `sklearn.datasets`)

- **Features (X):** 8 housing-related features including `MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
- **Target (y):** Median house price (`Price`)
- **Size:** 20,640 samples

## Project Structure

```
Regression.ipynb   # Main notebook
README.md
```

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Workflow

1. **Data Loading** — Fetch the California Housing dataset and convert to pandas DataFrames
2. **Null Check** — Verify there are no missing values in features or target
3. **Preprocessing** — Apply `StandardScaler` to normalize features and target (values vary at large scale)
4. **Train/Test Split** — 80/20 split with `random_state=42`
5. **Model Training & Evaluation** — Two models are trained and evaluated

## Models

### 1. Linear Regression (Single Feature)
Uses only `AveRooms` to predict `Price`.

- Finds the best-fit straight line (slope + intercept) through the training data
- Suitable here because the relationship between one feature and price is modeled as linear

### 2. Decision Tree Regressor (All Features)
Uses all 8 features to predict `Price`.

- Builds a tree structure that splits data based on feature conditions at each node
- Better suited for multi-feature prediction where relationships may be non-linear

## Evaluation Metrics

Both models are evaluated using:

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² Score | Coefficient of determination |

## Usage

Open the notebook in Jupyter and run all cells sequentially:

```bash
jupyter notebook Regression.ipynb
```

> **Note:** This project was built and tested using **Jupyter Notebook**. Make sure you have it installed:
> ```bash
> pip install notebook
> ```
