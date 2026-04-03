
import joblib
import pandas as pd
import numpy as np


# -------- load model artifacts once --------
model = joblib.load("Model/model.pkl")
explainer = joblib.load("Model/explainer.pkl")
feature_order = joblib.load("Model/feature_order.pkl")

def predict_url(features_df: pd.DataFrame) -> dict:
    """
    Takes a dataframe of extracted features (1 row)
    and returns prediction information.

    Returns:
        {
            prediction: "phishing" or "legitimate",
            probability: float,
            top_features: {feature: percentage}
        }
    """

    # Ensure correct feature order
    features_df = features_df[feature_order]

    # -------- model prediction --------
    prediction = model.predict(features_df)[0]

    probabilities = model.predict_proba(features_df)[0]
    probability = float(np.max(probabilities))

    # -------- SHAP explanation --------
    shap_values = explainer.shap_values(features_df)

    # For binary classification we use class 1


    ####shap_vals = shap_values[1][0]




    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]

    # 2. If SHAP returns a 3D array: (samples, features, classes) e.g., (1, 46, 2)
    elif len(shap_values.shape) == 3:
        # [first_sample, all_features, class_1]
        shap_vals = shap_values[0, :, 1]

    # 3. If SHAP returns a 2D array: (samples, features) e.g., (1, 46)
    elif len(shap_values.shape) == 2:
        # [first_sample]
        shap_vals = shap_values[0]

    else:
        # Fallback just in case
        shap_vals = shap_values[0]

    importance = pd.Series(
        np.abs(shap_vals),
        index=features_df.columns
    )




    # Top 3 features
    top3 = importance.sort_values(ascending=False).head(3)

    total_importance = importance.sum()

    top_features_percent = {
        feature: float((value / total_importance) * 100)
        for feature, value in top3.items()
    }

    # -------- format result --------
    result = {
        "prediction": "phishing" if prediction == 1 else "legitimate",
        "probability": probability,
        "top_features": top_features_percent
    }

    return result

