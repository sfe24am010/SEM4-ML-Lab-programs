# student_performance.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

# ---------- Step 1: Create a small dataset (Sample data) ----------
data = {
    'Quiz1': [7, 8, 6, 9, 5, 7, 8, 10, 9, 6],
    'Quiz2': [8, 7, 5, 9, 6, 8, 9, 10, 8, 7],
    'Internal1': [40, 42, 35, 45, 30, 38, 41, 48, 44, 36],
    'Internal2': [42, 40, 34, 46, 32, 39, 43, 47, 45, 37],
    'Assignment_Seminar': [4, 5, 3, 5, 3, 4, 4, 5, 5, 3],
    'Final_Score': [78, 82, 70, 90, 65, 76, 84, 95, 88, 72]
}

df = pd.DataFrame(data)

# ---------- Step 2: Prepare features ----------
X = df[['Quiz1', 'Quiz2', 'Internal1', 'Internal2', 'Assignment_Seminar']]
y_reg = df['Final_Score']

# ---------- Convert Regression Target to Classification ----------
# 1 = High performer (>=75), 0 = Low performer (<75)
y_class = (df['Final_Score'] >= 75).astype(int)

# ---------- Step 3: Train Linear Regression (for prediction) ----------
reg_model = LinearRegression()
reg_model.fit(X, y_reg)

# ---------- Step 4: Train Classification Model ----------
clf = LogisticRegression()
clf.fit(X, y_class)

# Predictions for accuracy & precision
y_pred_class = clf.predict(X)


# ---------- Step 5: Print Accuracy and Precision in % ----------
accuracy = accuracy_score(y_class, y_pred_class) * 100
precision = precision_score(y_class, y_pred_class) * 100

print(f"\nAccuracy Score: {accuracy:.2f}%")
print(f"Precision Score: {precision:.2f}%")

# ---------- Step 6: Get user input ----------
print("\n------ Predict Your Final Marks ------")
q1 = float(input("Enter Quiz 1 marks (out of 10): "))
q2 = float(input("Enter Quiz 2 marks (out of 10): "))
i1 = float(input("Enter Internal 1 marks (out of 50): "))
i2 = float(input("Enter Internal 2 marks (out of 50): "))
a_s = float(input("Enter Assignment + Seminar marks (out of 5): "))

# ---------- Step 7: Predict final score ----------
predicted_score = reg_model.predict([[q1, q2, i1, i2, a_s]])[0]

print(f"\nPredicted Final Exam Score: {predicted_score:.2f} / 100")

# ---------- Step 8: Plot comparison ----------
components = ['Quiz1', 'Quiz2', 'Internal1', 'Internal2', 'Assignment+Seminar', 'Predicted Final']
scores = [q1, q2, i1, i2, a_s, predicted_score]

plt.bar(components, scores, color=['skyblue', 'skyblue', 'orange', 'orange', 'green', 'red'])
plt.title('Student Performance Components vs Predicted Final Score')
plt.ylabel('Marks')
plt.ylim(0, 100)
plt.show()

