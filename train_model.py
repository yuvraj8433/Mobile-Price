import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
data = pd.read_csv("data/mobile_data.csv")

# ✅ Use only selected important features
selected_features = ["battery_power", "ram", "px_height", "px_width",
                     "four_g", "touch_screen", "wifi"]

X = data[selected_features]
y = data["price_range"]

# Check class balance
print("Class distribution:")
print(y.value_counts(), "\n")

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train the model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("✅ Model Performance on Test Set:\n")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High", "Very High"]))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Trained model saved at model/model.pkl")
