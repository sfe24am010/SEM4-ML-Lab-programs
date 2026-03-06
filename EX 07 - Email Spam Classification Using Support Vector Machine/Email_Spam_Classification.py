import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

# --------------------------------------------
# Step 1 — Load Dataset
# --------------------------------------------

data = pd.read_csv("spam_dataset.csv")

# Convert labels to numeric
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# --------------------------------------------
# Step 2 — Split Data
# --------------------------------------------

X = data['message']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# --------------------------------------------
# Step 3 — TF-IDF Vectorization
# --------------------------------------------

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --------------------------------------------
# Step 4 — Train SVM Model
# --------------------------------------------

svm_model = SVC(kernel='linear', probability=True)

svm_model.fit(X_train_vec, y_train)

# --------------------------------------------
# Step 5 — Test Prediction
# --------------------------------------------

y_pred = svm_model.predict(X_test_vec)

# --------------------------------------------
# Step 6 — Evaluation Metrics
# --------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
error = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("----------------------")
print(f"Accuracy :{accuracy*100:.2f}%")
print(f"Error Rate :{error*100:.2f}%")
print(f"Precision:{precision:.2f}")
print(f"Recall :{recall:.2f}")
print(f"F1 Score:{f1:.2f}")

print("\nConfusion Matrix:\n", cm)

# --------------------------------------------
# Step 7 — ROC Curve
# --------------------------------------------

y_prob = svm_model.predict_proba(X_test_vec)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM Spam Classifier")
plt.legend()

plt.show()

# --------------------------------------------
# Step 8 — USER INPUT PREDICTION
# --------------------------------------------

print("\nEMAIL SPAM DETECTOR")
print("----------------------")

user_email = input("Enter the Email Subject/Message: ")

# Transform input using same vectorizer
user_vec = vectorizer.transform([user_email])

prediction = svm_model.predict(user_vec)[0]

print("\nPrediction Result:")

if prediction == 1:
    print("This Email is SPAM")
else:
    print("This Email is NOT SPAM")

