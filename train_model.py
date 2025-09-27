import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
import os

# === Load dataset ===
cols = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]
# ضع ملف البيانات في فولدر data/ مع المشروع
data_file = os.path.join(os.path.dirname(__file__), "data", "processed.cleveland.data")

df = pd.read_csv(data_file, header=None, names=cols, na_values="?")

# Binary target
df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
df = df.drop(columns=["num"])

# Features & labels
X = df.drop("target", axis=1)
y = df["target"]

# Column groups
categorical_cols = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
numeric_cols = ["age","trestbps","chol","thalach","oldpeak"]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ]
)

# Final pipeline with best params
final_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, solver="liblinear", C=10, penalty="l1"))
])

# Train
final_pipeline.fit(X, y)

# Save model
save_path = os.path.join(os.path.dirname(__file__), "final_model.pkl")
joblib.dump(final_pipeline, save_path)
print("✅ Model trained and saved at:", save_path)
