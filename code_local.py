import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import pickle

# Reading Dataset
df = pd.read_csv("kdd_test.csv")
label_counts = df["labels"].value_counts()
valid_labels = label_counts[label_counts >= 2].index

df = df[df["labels"].isin(valid_labels)]

print("Remaining classes:", df["labels"].unique())
# Categorical columns
cat_cols = ["protocol_type", "service", "flag"]
# Encode categorical features
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
# Encode labels (target)
label_encoder = LabelEncoder()
df["labels"] = label_encoder.fit_transform(df["labels"])
X = df.drop(columns=["labels"])
y = df["labels"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = XGBClassifier(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",  # fast
)

model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

# Save model and encoders
with open("model_params.pkl", "wb") as f:
    pickle.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_encoders": encoders,
            "feature_names": X.columns.tolist(),
        },
        f,
    )

print("Model parameters saved to model_params.pkl")
print("Test Accuracy:", acc)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
