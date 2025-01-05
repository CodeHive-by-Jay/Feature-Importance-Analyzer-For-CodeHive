import pandas as pd
from data_processor import preprocess_data
from feature_importance import (
    compute_model_importance,
    plot_feature_importance,
    compute_shap_values,
    plot_shap_summary
)

def main():
    # Load the dataset
    print("Loading data...")
    dataset_path = "data/sample_dataset.csv"  # Replace with your dataset path
    data = pd.read_csv(dataset_path)

    # Preprocess the data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(data, target_column="target")

    # Compute feature importance using Random Forest
    print("Training model and computing feature importance...")
    model, feature_importance = compute_model_importance(X_train, y_train, model_type="random_forest")

    # Display and plot feature importance
    print("Feature importance:")
    print(feature_importance)
    print("Generating feature importance plot...")
    plot_feature_importance(feature_importance)

    # Compute SHAP values for explainability
    print("Computing SHAP values...")
    shap_values = compute_shap_values(model, X_train)

    # Generate SHAP summary plot
    print("Generating SHAP summary plot...")
    plot_shap_summary(shap_values, X_train)

if __name__ == "__main__":
    main()
