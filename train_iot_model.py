import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest


def build_iot_models(csv_path):
    print("1. Ingesting the ToN_IoT dataset...")
    # The low_memory flag prevents crashes when Pandas encounters mixed data types in large files
    df = pd.read_csv(csv_path, low_memory=False)

    print("2. Executing structural purging for IoT schema...")
    # We remove explicit identifiers and the granular 'type' column to prevent label leakage
    columns_to_drop = ['ts', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'type']
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Isolate the behavioral features (X) from the binary classification target (y)
    X = df.drop(columns=['label'])
    y = df['label']

    print("3. Dynamically defining the transformation matrix...")
    # We automatically separate categorical text strings from mathematical magnitudes
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    # High-Cardinality Safeguard: Drop text columns with excessive unique values
    valid_categorical = [col for col in categorical_features if X[col].nunique() < 50]
    dropped_cardinal = set(categorical_features) - set(valid_categorical)
    if dropped_cardinal:
        print(f" -> Dropping high-cardinality features: {dropped_cardinal}")
    X = X.drop(columns=list(dropped_cardinal))

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), numerical_features),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), valid_categorical)
        ])

    print("4. Partitioning data and applying mathematical normalization...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_transformed = preprocessor.fit_transform(X_train)

    print("5. Training the strict IoT Isolation Forest...")
    # IoT traffic is highly predictable. We use a 1% contamination rate for a tight baseline.
    iot_iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )
    iot_iso_forest.fit(X_train_transformed)

    print("6. Serializing IoT artifacts to disk...")
    # These exact filenames match what the realtime_engine.py expects when the IoT flag is enabled
    joblib.dump(preprocessor, 'iot_feature_transformer.pkl')
    joblib.dump(iot_iso_forest, 'iot_iso_forest.pkl')

    print("Phase 5 Training Complete! IoT artifacts are ready for segmented inference.")


if __name__ == "__main__":
    # Ensure the ToN_IoT CSV is in the correct directory before execution
    build_iot_models("Train_Test_Network.csv")