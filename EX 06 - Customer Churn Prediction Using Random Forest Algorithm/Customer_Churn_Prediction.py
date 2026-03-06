import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------- Step 1: Load Excel file ----------
df = pd.read_excel("customer_churn.xlsx")

# ---------- Step 2: Select features and target ----------
X = df[['CustomerAge', 'MonthlyCharges', 'Tenure',
        'ContractType', 'InternetService',
        'SupportCalls', 'TotalSpend']]

y = df['Churn']

# ---------- Step 3: Train Random Forest ----------
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X, y)

# ---------- Step 4: Prediction ----------
y_pred = model.predict(X)

# ---------- Step 5: Accuracy and Error ----------
accuracy = accuracy_score(y, y_pred)
error = 1 - accuracy

print("Accuracy:", round(accuracy, 2))
print("Error Rate:", round(error, 2))

# ---------- Step 6: Confusion Matrix ----------
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred))

# ---------- Step 7: User Input ----------
print("\n--- Customer Churn Prediction ---")
age = float(input("Customer Age: "))
mc = float(input("Monthly Charges: "))
tenure = float(input("Tenure (months): "))
contract = int(input("Contract Type (1=Long, 0=Monthly): "))
internet = int(input("Internet Service (1=Yes, 0=No): "))
calls = int(input("Support Calls: "))
spend = float(input("Total Spend: "))

result = model.predict([[age, mc, tenure, contract, internet, calls, spend]])

if result[0] == 1:
    print("Customer is likely to CHURN")
else:
    print("Customer is likely to STAY")

