# Linear Regression Temperature Prediction

This project performs a linear regression analysis on temperature data to predict Celsius from Fahrenheit. It includes statistical analysis to identify outliers and provides visualizations of the data and its residuals.

## Prerequisites

Make sure you have Python installed (Python 3.x is recommended). You will also need the following Python libraries:
- `pandas`
- `numpy`
- `statsmodels`
- `matplotlib`

## Getting Started

Follow these steps to set up and run the project:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/LinearRegressionPredictionC-F.git
   cd LinearRegressionPredictionC-F
   ```

2. **Install the required dependencies:**
   ```bash
   pip install pandas numpy statsmodels matplotlib
   ```

3. **Run the main script:**
   ```bash
   python main.py
   ```

## Usage

Once the script is running, it will:
1. Perform statistical analysis on the data.
2. Print the correlation (R-Squared) and list any identified outliers.
3. Display a window with two plots. **Note: You must close the plot window to continue.**
4. Prompt you to enter a Fahrenheit value to test the model's prediction.

## Analysis Steps

The script `main.py` is structured into the following key steps:

### 1. Data Ingestion & Cleaning
Reads the given dataset (`CitiesTemp.csv`). Since the raw data might contain strings with units (e.g., "100 °F"), the script extracts just the numerical values using regular expressions and converts them into floating-point numbers suitable for analysis.

### 2. Fit the Model
Uses Ordinary Least Squares (OLS) regression from the `statsmodels` library to fit a linear model to the data. This mathematically determines the relationship (the best-fitting slope and intercept) between Fahrenheit (the independent variable, X) and Celsius (the dependent variable, y).

### 3. Calculate Residuals & Z-Scores
Computes the residuals, which are the differences between the actual dataset values and the values predicted by our model. It then converts these residuals into Z-scores (standardized residuals) to measure how many standard deviations each point is away from the mean residual.

### 4. Identify Outliers
Applies a standard quantitative research threshold—specifically, any data point with an absolute Z-score greater than `2.0` (|Z| > 2)—to identify and flag significant outliers in the dataset.

### 5. Professional Visualization
Creates a professional figure with two subplots using `matplotlib`:
- **Plot A (Residual Distribution):** A histogram displaying the distribution of Z-scores, complete with vertical lines marking the defined outlier threshold.
- **Plot B (Scatter with Outlier Highlighting):** A scatter plot showing the original data points, drawing the calculated regression line through them, and clearly highlighting the previously identified outliers in red.

### 6. Prediction & Evaluation
After outputting the model's derived parameters (Slope and Intercept), the script enters an interactive loop. It asks the user to input a custom Fahrenheit temperature, applies the linear formula (`y = aX + b`) to predict the Celsius equivalent, and then calculates the exact prediction error by comparing it against the true mathematical conversion formula.