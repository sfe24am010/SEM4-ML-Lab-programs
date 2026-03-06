# simple_house_price_prediction.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------- Step 1: Load data from Excel ----------
# Make sure the Excel file is in the same folder as this script
df = pd.read_excel("house_data.xlsx")     # Change your excel file instead of this.

# ---------- Step 2: Separate input and output ----------
X = df[['Area_sqft']]   # independent variable
y = df['Price_Lakhs']   # dependent variable

# ---------- Step 3: Train the model ----------
model = LinearRegression()
model.fit(X, y)

# ---------- Step 4: Plot the regression line ----------
plt.scatter(X, y, label='Actual Data')
plt.plot(X, model.predict(X), label='Regression Line')
plt.xlabel("Area (sqft)")
plt.ylabel("Price (in Lakhs)")
plt.title("Simple Linear Regression - House Price Prediction")
plt.legend()
plt.show()

# ---------- Step 5: Take user input and predict ----------
print("------ House Price Prediction ------")
area = float(input("Enter the area of the house in sqft: "))

predicted_price = model.predict([[area]])[0]

print(f"\nEstimated House Price for {area} sqft = ₹{predicted_price:.2f} Lakhs")

# ---------- Step 6: Plot the user's input point ----------
plt.scatter(X, y, label='Training Data')
plt.plot(X, model.predict(X), label='Regression Line')
plt.scatter(area, predicted_price, s=100, label='Your Input Point')
plt.xlabel("Area (sqft)")
plt.ylabel("Price (in Lakhs)")
plt.title("Predicted Price Visualization")
plt.legend()
plt.show()

