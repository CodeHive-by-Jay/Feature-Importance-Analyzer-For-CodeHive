import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
import shap
import matplotlib.pyplot as plt

def compute_model_importance(X_train, y_train, model_type="random_forest"):
    """
    Computes feature importance based on the specified model type.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        model_type (str): Type of model to use. Options: 'random_forest', 'gradient_boosting', 'logistic_regression'.

    Returns:
        model: Trained model.
        feature_importance (pd.DataFrame): Feature importance scores.
    """
    # Initialize the model
    if model_type == "random_forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=42)
    elif model_type == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else:
        raise ValueError("Invalid model_type. Choose from 'random_forest', 'gradient_boosting', 'logistic_regression'.")

    # Train the model
    model.fit(X_train, y_train)

    # Compute feature importance
    if model_type in ["random_forest", "gradient_boosting"]:
        importance = model.feature_importances_
    elif model_type == "logistic_regression":
        importance = np.abs(model.coef_).flatten()

    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    return model, feature_importance

def plot_feature_importance(feature_importance):
    """
    Plots the feature importance scores.

    Parameters:
        feature_importance (pd.DataFrame): Feature importance scores.
    """
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance["Feature"], feature_importance["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()

def compute_shap_values(model, X_train):
    """
    Computes SHAP values for feature explainability.

    Parameters:
        model: Trained model.
        X_train (pd.DataFrame): Training features.

    Returns:
        shap_values: SHAP values for the dataset.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    return shap_values

def plot_shap_summary(shap_values, X_train):
    """
    Plots the SHAP summary plot for feature importance.

    Parameters:
        shap_values: SHAP values for the dataset.
        X_train (pd.DataFrame): Training features.
    """
    shap.summary_plot(shap_values, X_train)
