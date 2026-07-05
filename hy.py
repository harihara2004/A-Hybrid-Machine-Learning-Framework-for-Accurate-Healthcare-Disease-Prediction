"""
================================================================================
HYBRID HEALTHCARE DISEASE PREDICTION SYSTEM
================================================================================
Dataset      : Heart Disease Dataset (heart.csv)
Path         : D:\\project1\\heart.csv
Language     : Python 3.11
Libraries    : pandas, numpy, scikit-learn, matplotlib, seaborn, pickle

Description:
This project builds a Hybrid Ensemble Machine Learning system to predict
Heart Disease using Logistic Regression, Random Forest, and Gradient
Boosting classifiers combined using a Weighted Soft Voting Classifier.

The system also accepts manual patient input, predicts disease risk,
calculates risk percentage, and categorizes the patient into
Low / Moderate / High risk groups.
================================================================================
"""

# ==============================================================================
# SECTION 1: IMPORT LIBRARIES
# ==============================================================================
import pandas as pd                                   # Data handling
import numpy as np                                     # Numerical operations
import pickle                                           # Model serialization
import matplotlib.pyplot as plt                         # Plotting
import seaborn as sns                          # Advanced visualization
from sklearn.preprocessing import StandardScaler        # Feature scaling
from sklearn.feature_selection import RFE               # Recursive Feature Elimination
from sklearn.model_selection import train_test_split     # Train-test splitting
from sklearn.linear_model import LogisticRegression       # Logistic Regression model
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)                                                          # Ensemble models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)                                                          # Evaluation metrics

# Set a consistent visual style for all plots
sns.set(style="whitegrid")


# ==============================================================================
# SECTION 2: LOAD DATASET
# ==============================================================================
# Path to the dataset on the local Windows machine
DATA_PATH = r"D:\project1\heart.csv"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(DATA_PATH)

print("=" * 80)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 80)


# ==============================================================================
# SECTION 3: DATA EXPLORATION
# ==============================================================================
print("\n----- First 5 Rows of Dataset -----")
print(df.head())

print("\n----- Dataset Shape (Rows, Columns) -----")
print(df.shape)

print("\n----- Dataset Information -----")
print(df.info())

print("\n----- Statistical Summary -----")
print(df.describe())

print("\n----- Column Names -----")
print(df.columns.tolist())

print("\n----- Target Variable Distribution -----")
print(df['target'].value_counts())


# ==============================================================================
# SECTION 4: DATA PREPROCESSING
# ==============================================================================

# ---- 4.1 Check Missing Values ----
print("\n----- Missing Values in Each Column -----")
print(df.isnull().sum())

# Handle missing values if any exist (fill numeric columns with median)
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.median(numeric_only=True))
    print("\nMissing values were found and filled using median values.")
else:
    print("\nNo missing values found in the dataset.")

# ---- 4.2 Remove Duplicate Records ----
initial_rows = df.shape[0]
df = df.drop_duplicates()
final_rows = df.shape[0]
print(f"\nDuplicate records removed: {initial_rows - final_rows}")
print(f"Dataset shape after removing duplicates: {df.shape}")

# ---- 4.3 Separate Features and Target ----
# 'target' column: 1 = Heart Disease Present, 0 = No Heart Disease
X = df.drop('target', axis=1)          # Feature matrix (13 features)
y = df['target']                        # Target vector

feature_names = X.columns.tolist()
print(f"\nTotal Features Before Selection: {len(feature_names)}")
print("Feature Names:", feature_names)

# ---- 4.4 Feature Scaling using StandardScaler ----
# Standardize features to have mean=0 and standard deviation=1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

print("\nFeature scaling completed using StandardScaler.")


# ==============================================================================
# SECTION 5: FEATURE SELECTION (RFE - Recursive Feature Elimination)
# ==============================================================================
# Use Logistic Regression as the base estimator for RFE
rfe_estimator = LogisticRegression(max_iter=1000, random_state=2)

# Select top 10 most important features out of 13
NUM_FEATURES_TO_SELECT = 10
rfe_selector = RFE(estimator=rfe_estimator, n_features_to_select=NUM_FEATURES_TO_SELECT)
rfe_selector.fit(X_scaled, y)

# Get the mask of selected features
selected_mask = rfe_selector.support_
selected_features = X_scaled.columns[selected_mask].tolist()

print("\n----- Feature Selection using RFE -----")
print(f"Selected Features ({len(selected_features)}): {selected_features}")

# Reduce the feature matrix to only the selected features
X_selected = X_scaled[selected_features]


# ==============================================================================
# SECTION 6: TRAIN-TEST SPLIT
# ==============================================================================
# Split data into 80% training and 20% testing
# stratify=y ensures the same class ratio in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y,
    test_size=0.20,
    stratify=y,
    random_state=2
)

print("\n----- Train-Test Split -----")
print(f"Training Set Shape: {X_train.shape}")
print(f"Testing Set Shape : {X_test.shape}")


# ==============================================================================
# SECTION 7: HELPER FUNCTION FOR EVALUATION
# ==============================================================================
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Trains and evaluates a given model.
    Returns a dictionary of performance metrics.
    """
    # Predictions on train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Predicted probabilities for ROC-AUC (probability of class 1)
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba)
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)

    # Print results
    print(f"\n{'=' * 80}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 80}")
    print(f"Training Accuracy : {train_accuracy:.4f}")
    print(f"Testing Accuracy  : {test_accuracy:.4f}")
    print(f"Precision         : {precision:.4f}")
    print(f"Recall            : {recall:.4f}")
    print(f"F1-Score          : {f1:.4f}")
    print(f"ROC-AUC Score     : {roc_auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")

    return {
        "model_name": model_name,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "y_test_proba": y_test_proba
    }


# ==============================================================================
# SECTION 8: LOGISTIC REGRESSION
# ==============================================================================
log_reg_model = LogisticRegression(max_iter=1000, random_state=2)
log_reg_model.fit(X_train, y_train)
log_reg_results = evaluate_model(
    log_reg_model, "Logistic Regression", X_train, X_test, y_train, y_test
)


# ==============================================================================
# SECTION 9: RANDOM FOREST CLASSIFIER
# ==============================================================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=2)
rf_model.fit(X_train, y_train)
rf_results = evaluate_model(
    rf_model, "Random Forest Classifier", X_train, X_test, y_train, y_test
)


# ==============================================================================
# SECTION 10: GRADIENT BOOSTING CLASSIFIER
# ==============================================================================
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=2)
gb_model.fit(X_train, y_train)
gb_results = evaluate_model(
    gb_model, "Gradient Boosting Classifier", X_train, X_test, y_train, y_test
)


# ==============================================================================
# SECTION 11: HYBRID SOFT VOTING ENSEMBLE MODEL
# ==============================================================================
# Combine Logistic Regression, Random Forest, and Gradient Boosting
# using Weighted Soft Voting.
# Weights: Logistic Regression = 1, Random Forest = 2, Gradient Boosting = 3
hybrid_model = VotingClassifier(
    estimators=[
        ('lr', log_reg_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft',
    weights=[1, 2, 3]
)

# Train the Hybrid Model
hybrid_model.fit(X_train, y_train)

hybrid_results = evaluate_model(
    hybrid_model, "Hybrid Model (Weighted Soft Voting)", X_train, X_test, y_train, y_test
)


# ==============================================================================
# SECTION 12: PERFORMANCE EVALUATION - MODEL COMPARISON TABLE
# ==============================================================================
comparison_data = {
    "Model": [
        log_reg_results["model_name"],
        rf_results["model_name"],
        gb_results["model_name"],
        hybrid_results["model_name"]
    ],
    "Train Accuracy": [
        log_reg_results["train_accuracy"],
        rf_results["train_accuracy"],
        gb_results["train_accuracy"],
        hybrid_results["train_accuracy"]
    ],
    "Test Accuracy": [
        log_reg_results["test_accuracy"],
        rf_results["test_accuracy"],
        gb_results["test_accuracy"],
        hybrid_results["test_accuracy"]
    ],
    "Precision": [
        log_reg_results["precision"],
        rf_results["precision"],
        gb_results["precision"],
        hybrid_results["precision"]
    ],
    "Recall": [
        log_reg_results["recall"],
        rf_results["recall"],
        gb_results["recall"],
        hybrid_results["recall"]
    ],
    "F1-Score": [
        log_reg_results["f1_score"],
        rf_results["f1_score"],
        gb_results["f1_score"],
        hybrid_results["f1_score"]
    ],
    "ROC-AUC": [
        log_reg_results["roc_auc"],
        rf_results["roc_auc"],
        gb_results["roc_auc"],
        hybrid_results["roc_auc"]
    ]
}

comparison_df = pd.DataFrame(comparison_data)

print("\n" + "=" * 80)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 80)
print(comparison_df.to_string(index=False))


# ==============================================================================
# SECTION 13: VISUALIZATION
# ==============================================================================

# ---- 13.1 ROC Curve for All Models ----
plt.figure(figsize=(8, 6))

for result in [log_reg_results, rf_results, gb_results, hybrid_results]:
    fpr, tpr, _ = roc_curve(y_test, result["y_test_proba"])
    plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {result['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison - All Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_comparison.png")
plt.show()

# ---- 13.2 Accuracy Comparison Bar Chart ----
plt.figure(figsize=(8, 6))
sns.barplot(
    x="Model",
    y="Test Accuracy",
    data=comparison_df,
    palette="viridis"
)
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.title("Test Accuracy Comparison Across Models")
plt.ylabel("Test Accuracy")
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.show()

# ---- 13.3 Confusion Matrix Heatmaps ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
results_list = [log_reg_results, rf_results, gb_results, hybrid_results]

for ax, result in zip(axes.flatten(), results_list):
    sns.heatmap(
        result["confusion_matrix"],
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        cbar=False
    )
    ax.set_title(result["model_name"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrix_heatmaps.png")
plt.show()


# ==============================================================================
# SECTION 14: PATIENT PREDICTION (MANUAL INPUT)
# ==============================================================================
def get_patient_input():
    """
    Collects manual input from the user for all 13 Heart Disease features.
    Returns a pandas DataFrame with one row of patient data.
    """
    print("\n" + "=" * 80)
    print("ENTER PATIENT DETAILS FOR HEART DISEASE PREDICTION")
    print("=" * 80)

    age = float(input("Age (years): "))
    sex = float(input("Sex (1 = Male, 0 = Female): "))
    cp = float(input("Chest Pain Type (0-3): "))
    trestbps = float(input("Resting Blood Pressure (mm Hg): "))
    chol = float(input("Serum Cholesterol (mg/dl): "))
    fbs = float(input("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False): "))
    restecg = float(input("Resting ECG Results (0-2): "))
    thalach = float(input("Maximum Heart Rate Achieved: "))
    exang = float(input("Exercise Induced Angina (1 = Yes, 0 = No): "))
    oldpeak = float(input("ST Depression Induced by Exercise: "))
    slope = float(input("Slope of Peak Exercise ST Segment (0-2): "))
    ca = float(input("Number of Major Vessels Colored by Fluoroscopy (0-4): "))
    thal = float(input("Thalassemia (0-3): "))

    # Create a dictionary matching the original 13 features
    patient_dict = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    patient_df = pd.DataFrame([patient_dict])
    return patient_df


def predict_patient(patient_df, model, scaler, selected_features, all_feature_names):
    """
    Preprocesses patient input (scaling + feature selection),
    predicts heart disease, and calculates risk percentage.
    """
    # Ensure columns are in the same order as original training features
    patient_df = patient_df[all_feature_names]

    # Apply the same StandardScaler used during training
    patient_scaled = scaler.transform(patient_df)
    patient_scaled_df = pd.DataFrame(patient_scaled, columns=all_feature_names)

    # Select only the RFE-selected features
    patient_selected = patient_scaled_df[selected_features]

    # Predict class (0 = No Disease, 1 = Disease)
    prediction = model.predict(patient_selected)[0]

    # Predict probability of disease (class 1)
    probability = model.predict_proba(patient_selected)[0][1]

    return prediction, probability


# ==============================================================================
# SECTION 15: RISK PERCENTAGE AND CATEGORY
# ==============================================================================
def get_risk_category(probability):
    """
    Converts probability into a risk percentage and risk category.
    Low Risk     : 0% - 39%
    Moderate Risk: 40% - 69%
    High Risk    : 70% - 100%
    """
    risk_percentage = probability * 100

    if risk_percentage < 40:
        risk_category = "Low Risk"
    elif risk_percentage < 70:
        risk_category = "Moderate Risk"
    else:
        risk_category = "High Risk"

    return risk_percentage, risk_category


# ---- Run Prediction for a Manually Entered Patient ----
patient_data = get_patient_input()

prediction, probability = predict_patient(
    patient_data, hybrid_model, scaler, selected_features, feature_names
)

risk_percentage, risk_category = get_risk_category(probability)

print("\n" + "=" * 80)
print("PATIENT PREDICTION RESULT")
print("=" * 80)
print(f"Disease Risk Percentage : {risk_percentage:.2f}%")
print(f"Risk Category           : {risk_category}")

if prediction == 1:
    print("Final Prediction        : Heart Disease Detected")
else:
    print("Final Prediction        : No Heart Disease Detected")

print("=" * 80)


# ==============================================================================
# SECTION 16: MODEL SAVING
# ==============================================================================
# Save the trained Hybrid Model
with open("hybrid_model.pkl", "wb") as model_file:
    pickle.dump(hybrid_model, model_file)

# Save the fitted StandardScaler
with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

# Save the selected feature names (from RFE)
with open("selected_features.pkl", "wb") as features_file:
    pickle.dump(selected_features, features_file)

print("\n" + "=" * 80)
print("MODEL, SCALER, AND SELECTED FEATURES SAVED SUCCESSFULLY")
print("Files saved: hybrid_model.pkl, scaler.pkl, selected_features.pkl")
print("=" * 80)

# ==============================================================================
# END OF HYBRID HEALTHCARE DISEASE PREDICTION SYSTEM
# ==============================================================================


"""
================================================================================
HYBRID HEALTHCARE DISEASE PREDICTION SYSTEM - STREAMLIT FRONTEND
================================================================================
This app loads the pre-trained Hybrid Voting Model (hybrid_model.pkl),
the fitted StandardScaler (scaler.pkl), and the selected features
(selected_features.pkl) that were produced by hybrid_model.py.

Run this app with:
    streamlit run app.py

Make sure hybrid_model.pkl, scaler.pkl, and selected_features.pkl
are in the SAME FOLDER as this app.py file.
================================================================================
"""

