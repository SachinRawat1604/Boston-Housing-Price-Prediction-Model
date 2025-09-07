# Boston Housing Price Prediction Model

## Project Overview
This project focuses on predicting Boston housing prices using various features available in the dataset. The goal is to build a robust model that accurately estimates the median value of owner-occupied homes, providing valuable insights for real estate investment and valuation.

## Key Features

- **Data Exploration:** Comprehensive analysis of the Boston Housing dataset to understand feature distributions and relationships.
- **Predictive Modeling:** Implementation of machine learning algorithms to predict housing prices.
- **Model Evaluation:** Rigorous evaluation using appropriate metrics to ensure accuracy and reliability.

## Dataset Information

The dataset contains information on housing prices in the Boston area, with features such as:

- CRIM: Per capita crime rate by town
- ZN: Proportion of residential land zoned for large lots
- INDUS: Proportion of non-retail business acres per town
- CHAS: Charles River dummy variable
- NOX: Nitrogen oxides concentration
- RM: Average number of rooms per dwelling
- AGE: Proportion of old homes built prior to 1940
- DIS: Weighted distances to five Boston employment centers
- RAD: Index of accessibility to highways
- TAX: Property-tax rate per $10,000
- PTRATIO: Pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents
- LSTAT: Percentage of lower status of the population
- MEDV: Median value of owner-occupied homes in $1000's (target)

## Technologies Used

- **Programming Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/SachinRawat1604/Boston-Housing-Price-Prediction-Model.git
   cd Boston-Housing-Price-Prediction-Model
   ```

2. **Install dependencies:**
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

## Usage

1. **Data Preparation:** Load the `boston.csv` dataset into a pandas DataFrame.
2. **Exploratory Data Analysis (EDA):** Analyze the dataset to gain insights into feature distributions, correlations, and outliers.
3. **Feature Engineering (Optional):** Create new features or transform existing ones to improve model performance.
4. **Model Training:** Train a machine learning model (e.g., Linear Regression, Random Forest) using the prepared data.
5. **Model Evaluation:** Evaluate performance using metrics like Mean Squared Error (MSE) or R-squared.

### Example Usage (Conceptual)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("boston.csv")

# Prepare features (X) and target (y)
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

## Contributing

Fork the repo, create a new branch for your feature, make changes, and submit a pull request!

## License

This project is licensed under the MIT License.
