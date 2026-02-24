from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("CitiesTemp.csv")
F = df['Fahrenheit (°F)'].str.extract(r'(-?\d+)').astype(int)
C = df['Celsius (°C)'].str.extract(r'(-?\d+)').astype(int)

model = LinearRegression()
model.fit(F, C)

m = model.coef_[0][0]
b = model.intercept_[0]

print("\n--- Model Equation ---")
print(f"C = {m:.4f}F + ({b:.4f})")

plt.scatter(F, C, color='blue', label='Actual Data')
plt.plot(F, model.predict(F), color='red', linewidth=2, label='Linear Fit')
plt.xlabel('Fahrenheit (°F)')
plt.ylabel('Celsius (°C)')
plt.legend()
plt.show()