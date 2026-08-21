import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

# Suppress scikit-learn version warnings for cleaner output
warnings.filterwarnings('ignore')


def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    print(f"\n--- {model_name} Performance Metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"True Negatives (Safe):      {matrix[0][0]}")
    print(f"False Positives (Alarms):   {matrix[0][1]}")
    print(f"False Negatives (Misses):   {matrix[1][0]}")
    print(f"True Positives (Blocked):   {matrix[1][1]}")
    print("-" * 40)


def run_validation(csv_path):
    print("1. Loading Evaluation Dataset...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Sanitize column names
    df.columns = df.columns.str.strip().str.lower()

    print("2. Executing Schema Alignment...")
    # Map the testing set anomalies back to the training schema
    rename_map = {
        'smean': 'smeansz',
        'dmean': 'dmeansz',
        'response_body_len': 'res_bdy_len',
        'sinpkt': 'sintpkt',
        'dinpkt': 'dintpkt'
    }
    df = df.rename(columns=rename_map)

    print("3. Purging extraneous columns and cleaning data...")
    # Replace literal spaces and dashes with true NaNs
    df.replace(r'^\s+$', np.nan, regex=True, inplace=True)
    df.replace(['-', '?', ''], np.nan, inplace=True)

    # Drop columns the model has never seen
    columns_to_drop = ['id', 'attack_cat', 'rate']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    if 'label' not in df.columns:
        print("CRITICAL ERROR: 'label' column is missing from the testing set.")
        return

    X_test_raw = df.drop(columns=['label'])
    y_test_true = df['label'].astype(int)

    print("4. Loading Serialized AI Artifacts...")
    try:
        preprocessor = joblib.load('core_transformer.pkl')
        xgb_model = joblib.load('core_xgboost.pkl')
        iso_forest = joblib.load('core_iso_forest.pkl')
    except FileNotFoundError:
        print("CRITICAL ERROR: Model artifacts not found. Execute Phase 1 training first.")
        return

    print("5. Coercing testing data to match training matrix types...")
    # We must ensure the mathematical columns are strictly numeric to prevent crash errors
    numerical_cols = preprocessor.transformers_[0][2]
    for col in numerical_cols:
        if col in X_test_raw.columns:
            X_test_raw[col] = pd.to_numeric(X_test_raw[col], errors='coerce')

    print("6. Applying Mathematical Normalization...")
    # CRITICAL: We use .transform(), NOT .fit_transform(), to prevent data leakage
    X_test_scaled = preprocessor.transform(X_test_raw)

    print("7. Generating Predictions...")
    xgb_predictions = xgb_model.predict(X_test_scaled)

    iso_raw_predictions = iso_forest.predict(X_test_scaled)
    iso_mapped_predictions = [1 if x == -1 else 0 for x in iso_raw_predictions]

    evaluate_model(y_test_true, xgb_predictions, "Supervised Engine (XGBoost)")
    evaluate_model(y_test_true, iso_mapped_predictions, "Unsupervised Engine (Isolation Forest)")


if __name__ == "__main__":
    run_validation("UNSW_NB15_testing-set.csv")