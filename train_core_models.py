import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier


def build_offline_models(csv_path):
    print("1. Loading dataset...")
    df = pd.read_csv(csv_path, low_memory=False)

    # Sanitize column names
    df.columns = df.columns.str.strip().str.lower()

    # --- NEW FIX: Clean dirty network artifacts before processing ---
    print(" -> Cleaning dirty network artifacts (whitespace, dashes)...")
    # Replace literal spaces, dashes, and question marks with true NaNs
    df.replace(r'^\s+$', np.nan, regex=True, inplace=True)
    df.replace(['-', '?', ''], np.nan, inplace=True)

    if 'label' not in df.columns:
        if 'attack_cat' in df.columns:
            print(" -> Notice: 'label' column missing. Reconstructing from 'attack_cat'...")
            df['label'] = df['attack_cat'].notna().astype(int)
        else:
            print("CRITICAL ERROR: Both 'label' and 'attack_cat' are missing. Cannot proceed.")
            return

    print("2. Purging non-behavioral identifiers...")
    columns_to_drop = [
        'id', 'attack_cat',
        'srcip', 'dstip',
        'sport', 'dsport',
        'stime', 'ltime'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    X = df.drop(columns=['label'])
    y = df['label'].astype(int)

    print("3. Defining the transformation matrix...")
    categorical_cols = ['proto', 'service', 'state']
    categorical_cols = [c for c in categorical_cols if c in X.columns]
    numerical_cols = X.drop(columns=categorical_cols).columns.tolist()

    # --- NEW FIX: Force math columns to numeric, coercing any remaining garbage to NaN ---
    for col in numerical_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # --- NEW FIX: Build Imputation Pipelines ---
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', num_pipeline, numerical_cols),
            ('categorical', cat_pipeline, categorical_cols)
        ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("4. Normalizing the math (Z-Score & One-Hot with Imputation)...")
    X_train_transformed = preprocessor.fit_transform(X_train)

    print("5. Training XGBoost (Known Threats)...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    xgb_model.fit(X_train_transformed, y_train)

    print("6. Training Isolation Forest (Zero-Days)...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    iso_forest.fit(X_train_transformed)

    print("7. Serializing artifacts to disk...")
    joblib.dump(preprocessor, 'core_transformer.pkl')
    joblib.dump(xgb_model, 'core_xgboost.pkl')
    joblib.dump(iso_forest, 'core_iso_forest.pkl')

    print("Phase 1 Complete! Artifacts are safely written.")


if __name__ == "__main__":
    build_offline_models("UNSW_NB15_optimized.csv")