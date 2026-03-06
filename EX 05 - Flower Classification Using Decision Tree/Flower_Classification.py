import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ---------- Step 1: Load dataset ----------
iris = load_iris()

# Convert to DataFrame for clarity
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].replace({0:'setosa', 1:'versicolor', 2:'virginica'})

print("Sample Data:")
print(df.head())

# ---------- Step 2: Split into features and labels ----------
X = df[iris.feature_names]
y = df['species']

# ---------- Step 3: Train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# ---------- Step 4: Train Decision Tree model ----------
model = DecisionTreeClassifier(criterion='entropy', random_state=1)
model.fit(X_train, y_train)

# ---------- Step 5: Make predictions ----------
y_pred = model.predict(X_test)

# ---------- Step 6: Evaluate model ----------
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")

# ---------- Step 7: Visualize Decision Tree ----------
plt.figure(figsize=(12,8))
plot_tree(model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree for Iris Flower Classification")
plt.show()

# ---------- Step 8: Predict for user input ----------
print("\n------ Predict Flower Type ------")
sepal_length = float(input("Enter Sepal Length (cm): "))
sepal_width  = float(input("Enter Sepal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))
petal_width  = float(input("Enter Petal Width (cm): "))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
prediction = model.predict(input_data)[0]
print(f"\n The predicted flower species is: {prediction.capitalize()}")
