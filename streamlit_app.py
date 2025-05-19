import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Load model, scaler, features
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# SHAP explainer setup (tree explainer for RandomForest)
explainer = shap.Explainer(model)

# Streamlit interface
st.set_page_config(layout="wide")
st.title("🧠 Insurance Charges Prediction + SHAP Explanation")

st.write("Estimate charges and explain model output with SHAP values.")

# Input collection
age = st.slider("Age", 18, 100, 30)
sex = st.radio("Sex", ["Female", "Male"])
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.radio("Smoker?", ["No", "Yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Encode inputs
sex_val = 1 if sex == "Male" else 0
smoker_val = 1 if smoker == "Yes" else 0
region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0
bmi_smoker = bmi * smoker_val

# Prepare input
input_dict = {
    "age": age,
    "sex": sex_val,
    "bmi": bmi,
    "children": children,
    "smoker": smoker_val,
    "region_northwest": region_northwest,
    "region_southeast": region_southeast,
    "region_southwest": region_southwest,
    "bmi_smoker": bmi_smoker
}
input_df = pd.DataFrame([input_dict])[feature_columns]
input_scaled = scaler.transform(input_df)

# Predict and explain
if st.button("Predict Insurance Charges with SHAP"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"💰 Predicted Insurance Charge: **${prediction:,.2f}**")

    # SHAP values
    shap_values = explainer(input_df)

    # SHAP force plot
    st.subheader("🔍 SHAP Force Plot (Local Explanation)")
    fig_force = shap.plots.force(shap_values[0], matplotlib=True, show=False)
    st.pyplot(fig=plt.gcf())

    # SHAP summary bar plot
    st.subheader("📊 SHAP Summary (Global Feature Importance)")
    fig_summary, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig=fig_summary)

####################################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model and resources
model = joblib.load("random_forest_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")
scaler = joblib.load("scaler.pkl")
explainer = shap.Explainer(model)

st.title("📂 Batch Insurance Charge Prediction with SHAP")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])

if uploaded_file:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.write("✅ Uploaded Data Preview:")
        st.dataframe(df_input.head())

        # Required columns check
        required = ["age", "sex", "bmi", "children", "smoker", "region"]
        if not all(col in df_input.columns for col in required):
            st.error(f"❌ CSV must contain columns: {required}")
        else:
            # Encoding and feature engineering
            df_input["sex"] = df_input["sex"].map({"female": 0, "male": 1})
            df_input["smoker"] = df_input["smoker"].map({"no": 0, "yes": 1})
            df_input = pd.get_dummies(df_input, columns=["region"], drop_first=True)

            # Ensure all region columns are present
            for col in ["region_northwest", "region_southeast", "region_southwest"]:
                if col not in df_input.columns:
                    df_input[col] = 0

            df_input["bmi_smoker"] = df_input["bmi"] * df_input["smoker"]

            # Reorder and scale
            X = df_input[feature_columns]
            X_scaled = scaler.transform(X)

            # Predict
            predictions = model.predict(X_scaled)
            df_input["Predicted Charges"] = predictions

            st.subheader("📊 Prediction Results")
            st.dataframe(df_input[["age", "bmi", "smoker", "children", "Predicted Charges"]])

            # SHAP summary plot
            st.subheader("📈 SHAP Summary for Batch")
            shap_values = explainer(X)
            fig_summary, ax = plt.subplots()
            shap.plots.bar(shap_values, show=False)
            st.pyplot(fig=fig_summary)

            # Optional: Download predictions
            csv_output = df_input.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions as CSV", data=csv_output, file_name="predicted_charges.csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
