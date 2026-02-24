import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 1. Data Ingestion & Cleaning
df = pd.read_csv("CitiesTemp.csv")
X = df['Fahrenheit (°F)'].str.extract(r'(-?\d+)').astype(float)
y = df['Celsius (°C)'].str.extract(r'(-?\d+)').astype(float)

# 2. Fit the Model (Finding the Correlation/Relationship)
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

# 3. Calculate Residuals & Z-Scores
df['Residual'] = model.resid
df['Z_Score'] = (df['Residual'] - df['Residual'].mean()) / df['Residual'].std()

# 4. Identify Outliers (Threshold of |Z| > 2 is common in Quant research)
threshold = 2.0
outliers = df[df['Z_Score'].abs() > threshold]

print(f"--- Statistical Analysis ---")
print(f"Correlation (R-Squared): {model.rsquared:.4f}")
print(f"Number of Outliers detected: {len(outliers)}")
print("\n--- Outlier Cities ---")
print(outliers[['Fahrenheit (°F)', 'Celsius (°C)', 'Z_Score']])

# 5. Professional Visualization
plt.figure(figsize=(12, 6))

# Plot A: Residual Distribution
plt.subplot(1, 2, 1)
plt.hist(df['Z_Score'], bins=15, color='skyblue', edgecolor='black')
plt.axvline(threshold, color='red', linestyle='--', label='Outlier Threshold')
plt.axvline(-threshold, color='red', linestyle='--')
plt.title("Distribution of Z-Scores (Standardized Residuals)")
plt.xlabel("Z-Score")
plt.legend()

# Plot B: Scatter with Outlier Highlighting
plt.subplot(1, 2, 2)
plt.scatter(X, y, color='blue', alpha=0.5, label='In-Sample Data')
plt.scatter(outliers.iloc[:, 0], outliers.iloc[:, 1], color='red', label='Significant Outliers')
plt.plot(X, model.predict(X_const), color='black', label='Regression Line')
plt.title("F vs C: Regression with Outlier Detection")
plt.legend()

plt.tight_layout()
plt.show()

a = model.params.iloc[1]  # Slope
b = model.params.iloc[0]  # Intercept
print(f"\n--- Model Parameters ---")
print(f"Slope: {a:.4f}")
print(f"Intercept: {b:.4f}")

# 6. Prediction & Evaluation
while True:
    try:
        input_temp_f = int(input("Enter temperature in Fahrenheit: "))
        break
    except ValueError:
        print("Invalid input. Please enter an integer.")
predicted_temp_c = a * input_temp_f + b

print(f"\n--- Prediction ---")
print(f"Predicted Celsius for {input_temp_f}°F: {predicted_temp_c:.2f}°C")

# Calculate the error for this prediction
actual_temp_c = (input_temp_f - 32) * 5/9
error = predicted_temp_c - actual_temp_c

print(f"Actual Celsius for {input_temp_f}°F: {actual_temp_c:.2f}°C")
print(f"Prediction Error: {error:.2f}°C")