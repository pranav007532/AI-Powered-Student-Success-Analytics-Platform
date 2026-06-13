import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ==================================================
# LOAD DATASET
# ==================================================

print("\nLoading dataset...")

data = pd.read_csv("Student_Performance.csv")

# Convert Yes/No to 1/0

data["Extracurricular Activities"] = data[
    "Extracurricular Activities"
].map({
    "Yes": 1,
    "No": 0
})

print("Dataset Loaded Successfully!")

# ==================================================
# FEATURES & TARGET
# ==================================================

X = data[
    [
        "Hours Studied",
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced"
    ]
]

y = data["Performance Index"]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==================================================
# MODELS
# ==================================================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

scores = {}

print("\n===== MODEL COMPARISON =====\n")

# ==================================================
# TRAIN ALL MODELS
# ==================================================

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    score = r2_score(
        y_test,
        predictions
    )

    scores[name] = score

    print(f"{name}: {score:.4f}")

# ==================================================
# SAVE MODEL COMPARISON
# ==================================================

comparison_df = pd.DataFrame({
    "Model": list(scores.keys()),
    "R2 Score": list(scores.values())
})

comparison_df.to_csv(
    "model_scores.csv",
    index=False
)

print("\nModel scores saved!")

# ==================================================
# BEST MODEL
# ==================================================

best_model_name = max(
    scores,
    key=scores.get
)

best_model = models[
    best_model_name
]

best_score = scores[
    best_model_name
]

print("\n==============================")
print(f"Best Model : {best_model_name}")
print(f"Accuracy   : {best_score:.4f}")
print("==============================")

# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(
    best_model,
    "student_model.pkl"
)

print("\nBest model saved successfully!")

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("\nFeature importance saved!")

# ==================================================
# DATASET SUMMARY
# ==================================================

print("\n===== DATASET INFO =====")

print(f"Rows    : {len(data)}")
print(f"Columns : {len(data.columns)}")

print("\nDataset Columns:")

for column in data.columns:
    print("-", column)

print("\nTraining Complete!")