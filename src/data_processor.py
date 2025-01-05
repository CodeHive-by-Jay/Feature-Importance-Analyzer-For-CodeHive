import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_dataset(filepath):
    """
    Loads a dataset from a CSV file.
    """
    try:
        data = pd.read_csv(filepath)
        return data
    except Exception as e:
        raise FileNotFoundError(f"Error loading dataset: {e}")

def preprocess_data(data, target_column):
    """
    Preprocesses the dataset:
    - Handles missing values.
    - Encodes categorical features.
    - Scales numerical features.
    - Splits into train/test sets.
    
    Parameters:
        data (pd.DataFrame): The dataset.
        target_column (str): The name of the target column.

    Returns:
        X_train, X_test, y_train, y_test: Preprocessed train/test sets.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    # Encode categorical features
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    
    # Scale numerical features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
