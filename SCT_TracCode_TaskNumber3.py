import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# ==============================
# CONFIG
# ==============================
CAT_DIR = r"C:\Users\Dell\OneDrive\Documents\archive\PetImages\Cat"   # <-- change this path
DOG_DIR = r"C:\Users\Dell\Downloads\archive (2)\archive\PetImages\Dog"
IMG_SIZE = 64                   # Resize images to 64x64

# ==============================
# LOAD DATASET
# ==============================ś
def load_data(cat_dir, dog_dir, img_size):
    data, labels = [], []

    # Load cat images
    for file in os.listdir(cat_dir)[:1000]:   # use 1000 for quick training
        img = cv2.imread(os.path.join(cat_dir, file))
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(gray.flatten())
            labels.append(0)   # 0 = cat

    # Load dog images
    for file in os.listdir(dog_dir)[:1000]:
        img = cv2.imread(os.path.join(dog_dir, file))
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            data.append(gray.flatten())
            labels.append(1)   # 1 = dog

    return np.array(data), np.array(labels)

print("📂 Loading dataset...")
X, y = load_data(CAT_DIR, DOG_DIR, IMG_SIZE)

# ==============================
# SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# TRAIN SVM
# ==============================
print("⚡ Training SVM classifier...")
svm = SVC(kernel="linear", probability=True)  # linear kernel SVM
svm.fit(X_train, y_train)

# ==============================
# EVALUATE
# ==============================
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {acc*100:.2f}%")

# ==============================
# CLASSIFY NEW IMAGE
# ==============================
def classify_image(img_path, model, img_size):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Error: Image not found!")
        return
    img = cv2.resize(img, (img_size, img_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = gray.flatten().reshape(1, -1)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    label = "Cat 🐱" if pred == 0 else "Dog 🐶"
    print(f"🔍 Prediction: {label} (Confidence: {max(prob)*100:.2f}%)")

# ==============================
# TEST WITH NEW IMAGE
# ==============================
test_image_path = r"C:\Users\Dell\Downloads\pexels-valeriya-1805164.jpg"   # <-- change this to any cat/dog image
classify_image(test_image_path, svm, IMG_SIZE)
