import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ------------------------------
# Load Models + Preprocessor
# ------------------------------
@st.cache_resource
def load_models():
    # Try loading RF model with fallback names
    rf_model = None
    for name in ["best_random_forest.joblib", "best_model_random_forest.joblib", "rf_model.joblib"]:
        try:
            rf_model = joblib.load(name)
            break
        except (FileNotFoundError, Exception):
            continue
    
    # Try loading LR model with fallback names
    lr_model = None
    for name in ["logistic_regression.joblib", "logistic_regression_model.joblib"]:
        try:
            lr_model = joblib.load(name)
            break
        except (FileNotFoundError, Exception):
            continue
    
    # Try loading preprocessor with fallback names
    preprocessor = None
    for name in ["preprocessor.joblib", "scaler.joblib"]:
        try:
            preprocessor = joblib.load(name)
            break
        except (FileNotFoundError, Exception):
            continue
    
    # Neural Network is optional (TensorFlow import removed for faster loading)
    nn_model = None

    return rf_model, lr_model, nn_model, preprocessor


rf_model, lr_model, nn_model, preprocessor = load_models()

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.markdown("""
# ❤️ Heart Disease Prediction App

Enter medical details below, and the system will predict whether the patient has **heart disease**.

Powered by:
- **Random Forest**
- **Logistic Regression**
- **Neural Network (optional)**

---
""")

# ----------------------------------
# User Input Form
# ----------------------------------
st.subheader("🧍 Patient Medical Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp_options = {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}
    cp = st.selectbox("Chest Pain Type (cp)", options=list(cp_options.keys()), format_func=lambda x: f"{x} — {cp_options[x]}")
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 120)
    chol = st.number_input("Cholesterol Level (chol)", 100, 600, 240)
    fbs_options = {0: "<= 120 mg/dl", 1: "> 120 mg/dl"}
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (fbs)", options=list(fbs_options.keys()), format_func=lambda x: f"{x} — {fbs_options[x]}")

with col2:
    restecg_options = {0: "Normal", 1: "ST-T abnormality", 2: "Left ventricular hypertrophy"}
    restecg = st.selectbox("Resting ECG (restecg)", options=list(restecg_options.keys()), format_func=lambda x: f"{x} — {restecg_options[x]}")
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", 60, 250, 150)
    exang_options = {0: "No", 1: "Yes"}
    exang = st.selectbox("Exercise Induced Angina (exang)", options=list(exang_options.keys()), format_func=lambda x: f"{x} — {exang_options[x]}")
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
    slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    slope = st.selectbox("Slope (slope)", options=list(slope_options.keys()), format_func=lambda x: f"{x} — {slope_options[x]}")
    ca_options = {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels", 4: "4 vessels"}
    ca = st.selectbox("No. of Major Vessels (ca)", options=list(ca_options.keys()), format_func=lambda x: f"{x} — {ca_options[x]}")
    thal_options = {0: "Unknown", 1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}
    thal = st.selectbox("Thalassemia (thal)", options=list(thal_options.keys()), format_func=lambda x: f"{x} — {thal_options[x]}")

# ----------------------------------
# Prepare input data
# ----------------------------------
def prepare_input():
    sex_num = 1 if sex == "Male" else 0
    return pd.DataFrame([{
        "age": age,
        "sex": sex_num,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

def one_hot_encode_input(df):
    """One-hot-encode categorical columns to match RF model's expected format.
    
    RF model expects: cp_1, cp_2, cp_3 (3 dummies from 4 categories)
                     thal_1, thal_2, thal_3 (3 dummies from 4 categories)
                     slope_1, slope_2 (2 dummies from 3 categories)
    """
    df_encoded = df.copy()
    
    # One-hot-encode with drop_first=True for most to match RF training
    # cp: 0,1,2,3 -> cp_1, cp_2, cp_3 (drop cp_0)
    cp_dummies = pd.get_dummies(df_encoded[['cp']], prefix='cp', drop_first=True)
    df_encoded = df_encoded.drop('cp', axis=1)
    df_encoded = pd.concat([df_encoded, cp_dummies], axis=1)
    
    # thal: 0,1,2,3 -> thal_1, thal_2, thal_3 (drop thal_0)
    thal_dummies = pd.get_dummies(df_encoded[['thal']], prefix='thal', drop_first=True)
    df_encoded = df_encoded.drop('thal', axis=1)
    df_encoded = pd.concat([df_encoded, thal_dummies], axis=1)
    
    # slope: 0,1,2 -> slope_1, slope_2 (drop slope_0)
    slope_dummies = pd.get_dummies(df_encoded[['slope']], prefix='slope', drop_first=True)
    df_encoded = df_encoded.drop('slope', axis=1)
    df_encoded = pd.concat([df_encoded, slope_dummies], axis=1)
    
    # Other categorical columns: ca, exang, fbs, restecg (binary or will drop one)
    for col in ['ca', 'exang', 'fbs', 'restecg']:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[[col]], prefix=col, drop_first=False)
            df_encoded = df_encoded.drop(col, axis=1)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
    
    return df_encoded

# ----------------------------------
# Prediction
# ----------------------------------
st.markdown("---")
st.subheader("🔍 Prediction Results")

# Initialize session state for prediction results
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'prediction_failed' not in st.session_state:
    st.session_state['prediction_failed'] = False

if st.button("Predict Heart Disease"):
    df_input = prepare_input()
    
    # Check if required models/preprocessor are loaded
    if rf_model is None or preprocessor is None:
        st.error("❌ Required models not loaded. Check messages above.")
        st.session_state['prediction_failed'] = True
        st.session_state['prediction_result'] = None
    else:
        try:
            # Separate numeric and categorical features
            numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            df_numeric = df_input[numeric_cols]
            
            # Scale numeric features using preprocessor
            df_numeric_scaled = preprocessor.transform(df_numeric)
            df_numeric_scaled = pd.DataFrame(df_numeric_scaled, columns=numeric_cols)
            
            # One-hot-encode categorical features
            df_encoded = one_hot_encode_input(df_input)
            
            # Drop numeric columns from encoded (already scaled)
            df_encoded_cats = df_encoded.drop(columns=numeric_cols, errors='ignore')
            
            # Combine scaled numeric + encoded categorical
            X_processed = pd.concat([df_numeric_scaled, df_encoded_cats], axis=1)
            
            # Ensure column order matches RF model's expectations
            expected_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca', 'cp_1', 'cp_2', 'cp_3', 'thal_1', 'thal_2', 'thal_3', 'slope_1', 'slope_2']
            for col in expected_cols:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            X_processed = X_processed[expected_cols]
        except Exception as e:
            st.error("❌ Preprocessing Failed")
            st.error(str(e))
            st.session_state['prediction_failed'] = True
            st.session_state['prediction_result'] = None
        else:
            try:
                # Random Forest Prediction
                rf_pred = rf_model.predict(X_processed)[0]
                rf_prob = rf_model.predict_proba(X_processed)[:, 1][0]

                # Logistic Regression Prediction
                lr_pred = lr_model.predict(X_processed)[0] if lr_model else None
                lr_prob = lr_model.predict_proba(X_processed)[:, 1][0] if lr_model else None

                # Neural Network Prediction (Optional)
                if nn_model:
                    nn_prob = nn_model.predict(X_processed).ravel()[0]
                    nn_pred = int(nn_prob > 0.5)
                else:
                    nn_prob = None
                    nn_pred = None

                # Store in session state
                st.session_state['prediction_result'] = {
                    'rf_pred': int(rf_pred),
                    'rf_prob': float(rf_prob),
                    'lr_pred': int(lr_pred) if lr_pred is not None else None,
                    'lr_prob': float(lr_prob) if lr_prob is not None else None,
                    'nn_pred': int(nn_pred) if nn_pred is not None else None,
                    'nn_prob': float(nn_prob) if nn_prob is not None else None,
                }
                st.session_state['prediction_failed'] = False

                # Display Results
                st.write("### 📌 Model Predictions")
                st.write(f"**Random Forest:** {'🔴 Disease Likely' if rf_pred == 1 else '🟢 No Disease'}  (Probability: {rf_prob:.2f})")
                if lr_prob is not None:
                    st.write(f"**Logistic Regression:** {'🔴 Disease Likely' if lr_pred == 1 else '🟢 No Disease'}  (Probability: {lr_prob:.2f})")

                if nn_model and nn_prob is not None:
                    st.write(f"**Neural Network:** {'🔴 Disease Likely' if nn_pred == 1 else '🟢 No Disease'}  (Probability: {nn_prob:.2f})")

                # Final conclusion (Based on RF)
                st.markdown("---")
                st.subheader("🧠 Final Conclusion (Best Model: Random Forest)")

                if rf_pred == 1:
                    st.error("⚠️ High likelihood of heart disease.\nPlease consult a cardiologist.")
                else:
                    st.success("✅ No major signs of heart disease.\nStay healthy and monitor regularly!")

                st.markdown("---")
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.session_state['prediction_failed'] = True
                st.session_state['prediction_result'] = None

# Show decision button - persistent across reruns
if st.button("Show Decision (0/1)"):
    if st.session_state['prediction_failed'] or st.session_state['prediction_result'] is None:
        st.warning("Decision not available because prediction failed.")
    else:
        pred_result = st.session_state['prediction_result']
        rf_pred = pred_result.get('rf_pred')
        label = 1 if rf_pred == 1 else 0
        st.write(f"**Predicted label (0 = No disease, 1 = Disease):** {label}")
   