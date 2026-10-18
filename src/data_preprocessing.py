import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load the raw insurance dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Standarized categorical text and remove exact duplicate rows."""
    df = df.copy()

    # Standardize column names and categorical text
    df.columns = df.columns.str.strip().str.lower()

    categorical_columns = ['sex', 'smoker', 'region']

    for column in categorical_columns:
        df[column] = df[column].str.strip().str.lower()

    # Check for missing values
    if df.isnull().any().any():
        missing_counts = df.isnull().sum()
        raise ValueError(
            f"Missing values found:\n{missing_counts}"
        )

    # Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    return df

def split_data(df, target_column="charges"):
    """Split the dataset into training, validation, and test sets."""

    X = df.drop(columns=[target_column])
    y = df[[target_column]]

    # 70% training, 30% temporary split for validation and test
    X_train, X_temp, y_train, y_temp = train_test_split (
        X,
        y,
        test_size=0.30,
        random_state=42
    )

    #Split the remaining 30% equally into validating and test suites
    X_val, X_test, y_val, y_test = train_test_split (
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42
    )

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )

def encode_categorical_features(X_train, X_val, X_test):
    """Encode categorical features consistently across all splits"""

    datasets = [
        X_train.copy(),
        X_val.copy(),
        X_test.copy()
    ]

    # Binary encoding
    binary_mapping = {
        "sex" : {"female" : 0, "male" : 1},
        "smoker" : {"no" : 0, "yes" : 1}
    }

    for df in datasets:
        for column, mapping in binary_mapping.items():
            unknown_values = set(df[column].dropna().unique()) - set(mapping)

            if unknown_values:
                raise ValueError(
                    f"Unexpected values in {column} : {unknown_values}"
                )

            df[column] = df[column].map(mapping).astype("int64")

    # Use a fixed category order for consistent region encoding
    regions = ["northeast", "northwest", "southeast", "southwest"]

    for df in datasets:
        unknown_regions = set(df["region"].dropna().unique()) - set(regions)

        if unknown_regions:
            raise ValueError(
                f"Unknown region values : {unknown_regions}"
            )

        region_dtype = pd.CategoricalDtype(categories=regions)

        region_categories = df["region"].astype(region_dtype)

        region_dummies = pd.get_dummies(
            region_categories,
            prefix="region",
            drop_first=True,
            dtype=int
        )

        df.drop(columns=['region'], inplace=True)

        df[region_dummies.columns] = region_dummies

    X_train_encoded, X_val_encoded, X_test_encoded = datasets

    return X_train_encoded, X_val_encoded, X_test_encoded

def scale_continuous_features(X_train, X_val, X_test):
    """Standardize continuous features using training statistics only."""

    continuous_features = ["age", "bmi", "children"]

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    scaler = StandardScaler()

    # Learn mean and standard deviation from training data only
    X_train[continuous_features] = scaler.fit_transform(
        X_train[continuous_features]
    )

    # Reuse the training fitted scaler
    X_val[continuous_features] = scaler.transform(
        X_val[continuous_features]
    )

    X_test[continuous_features] = scaler.transform(
        X_test[continuous_features]
    )

    return X_train, X_val, X_test, scaler

def convert_to_numpy(X_train, X_val, X_test, y_train, y_val, y_test):
    """Convert processed Dataframes into numpy arrays."""

    X_train = X_train.to_numpy(dtype=np.float64)
    X_val = X_val.to_numpy(dtype=np.float64)
    X_test = X_test.to_numpy(dtype=np.float64)

    Y_train = y_train.to_numpy(dtype=np.float64)
    Y_val = y_val.to_numpy(dtype=np.float64)
    Y_test = y_test.to_numpy(dtype=np.float64)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test