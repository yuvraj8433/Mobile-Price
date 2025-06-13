import pandas as pd
import numpy as np
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset
data = pd.read_csv("data/mobile_data.csv")

# Split into features and target
X = data.drop("price_range", axis=1)
y = data["price_range"]

# Check class balance
print("Class distribution:")
print(y.value_counts(), "\n")

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train a more powerful model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("✅ Model Performance on Test Set:\n")
print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High", "Very High"]))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the model
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Trained model saved at model/model.pkl")
