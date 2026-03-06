# handwritten_digit_fixed_knn.py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ---------- Step 1: Train the model ----------
digits = load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model trained with accuracy: {acc*100:.2f}%")
# ---------- Step 2: Load your handwritten digit image ----------
image_path = input("\nEnter your handwritten digit image filename (digit.png): ")
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: Could not load image. Please check your path.")
    exit()
# ---------- Step 3: Preprocess the image ----------
# Resize to 8x8 (same as dataset)
img_resized = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
# Determine if background is lighter or darker and invert if needed
if np.mean(img_resized)> 127:
 img_resized = 255 - img_resized
# Apply binary thresholding for clarity
_, img_thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Scale pixel values to 0–16 (dataset range)
img_scaled = (img_thresh / 16).astype(np.float32)

# Flatten for model input
input_data = img_scaled.flatten().reshape(1, -1)
# ---------- Step 4: Predict ----------
predicted_digit = model.predict(input_data)[0]
# ---------- Step 5: Show results ----------
print("\nPredicted Digit: {predicted_digit}")
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(img_thresh, cmap="gray")
plt.title("Predicted Digit: {predicted_digit}")
plt.axis("off")
plt.tight_layout()
plt.show()