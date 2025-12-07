import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient data to predict the likelihood of heart disease.")

# -------------------------------
# Load models and preprocessor
# -------------------------------
@st.cache_resource
def load_models():
    # Try loading preprocessor with fallback names
    preprocessor = None
    for name in ["preprocessor.joblib", "scaler.joblib"]:
        try:
            preprocessor = joblib.load(name)
            st.success(f"✅ Loaded preprocessor: {name}")
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Found {name} but failed to load: {e}")
            continue
    
    if preprocessor is None:
        st.error("❌ Preprocessor not found. Expected 'preprocessor.joblib' or 'scaler.joblib'")
    
    # Try loading RF model with fallback names
    rf_model = None
    for name in ["best_random_forest.joblib", "best_model_random_forest.joblib", "rf_model.joblib"]:
        try:
            rf_model = joblib.load(name)
            st.success(f"✅ Loaded RF model: {name}")
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Found {name} but failed to load: {e}")
            continue
    
    if rf_model is None:
        st.error("❌ Random Forest model not found")
    
    # Try loading LR model with fallback names
    lr_model = None
    for name in ["logistic_regression.joblib", "logistic_regression_model.joblib"]:
        try:
            lr_model = joblib.load(name)
            st.success(f"✅ Loaded LR model: {name}")
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"Found {name} but failed to load: {e}")
            continue
    
    if lr_model is None:
        pass  # Logistic Regression is optional, don't warn

    nn_model = None  # Neural Network loading removed (requires tensorflow)

    return preprocessor, rf_model, lr_model, nn_model


preprocessor, rf_model, lr_model, nn_model = load_models()

# ----------------------------------
# Input fields
# ----------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp_options = {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}
    cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: f"{x} — {cp_options[x]}")
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 240)
    fbs_options = {0: "<= 120 mg/dl", 1: "> 120 mg/dl"}
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl", options=list(fbs_options.keys()), format_func=lambda x: f"{x} — {fbs_options[x]}")

with col2:
    restecg_options = {0: "Normal", 1: "ST-T abnormality", 2: "Left ventricular hypertrophy"}
    restecg = st.selectbox("Resting ECG", options=list(restecg_options.keys()), format_func=lambda x: f"{x} — {restecg_options[x]}")
    thalach = st.number_input("Max Heart Rate", 60, 250, 150)
    exang_options = {0: "No", 1: "Yes"}
    exang = st.selectbox("Exercise Induced Angina", options=list(exang_options.keys()), format_func=lambda x: f"{x} — {exang_options[x]}")
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    slope_options = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    slope = st.selectbox("Slope", options=list(slope_options.keys()), format_func=lambda x: f"{x} — {slope_options[x]}")
    ca_options = {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels", 4: "4 vessels"}
    ca = st.selectbox("Major Vessels", options=list(ca_options.keys()), format_func=lambda x: f"{x} — {ca_options[x]}")
    thal_options = {0: "Unknown", 1: "Normal", 2: "Fixed defect", 3: "Reversible defect"}
    thal = st.selectbox("Thalassemia", options=list(thal_options.keys()), format_func=lambda x: f"{x} — {thal_options[x]}")

# ----------------------------------
# Prepare input
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
# Initialize session state for prediction results
if 'prediction_result' not in st.session_state:
    st.session_state['prediction_result'] = None
if 'prediction_failed' not in st.session_state:
    st.session_state['prediction_failed'] = False

if st.button("Predict"):
    if preprocessor is None or rf_model is None:
        st.error("❌ Required models not loaded. Check messages above.")
        st.session_state['prediction_failed'] = True
        st.session_state['prediction_result'] = None
    else:
        df = prepare_input()
        
        try:
            # Separate numeric and categorical features
            numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            df_numeric = df[numeric_cols]
            
            # Scale numeric features using preprocessor
            df_numeric_scaled = preprocessor.transform(df_numeric)
            df_numeric_scaled = pd.DataFrame(df_numeric_scaled, columns=numeric_cols)
            
            # One-hot-encode categorical features
            df_encoded = one_hot_encode_input(df)
            
            # Drop numeric columns from encoded (already scaled)
            df_encoded_cats = df_encoded.drop(columns=numeric_cols, errors='ignore')
            
            # Combine scaled numeric + encoded categorical
            X = pd.concat([df_numeric_scaled, df_encoded_cats], axis=1)
            
            # Ensure column order matches RF model's expectations
            expected_cols = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'ca', 'cp_1', 'cp_2', 'cp_3', 'thal_1', 'thal_2', 'thal_3', 'slope_1', 'slope_2']
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0
            X = X[expected_cols]
        except Exception as e:
            st.error("❌ Preprocessing failed")
            st.error(str(e))
            st.session_state['prediction_failed'] = True
            st.session_state['prediction_result'] = None
        else:
            try:
                rf_prob = rf_model.predict_proba(X)[:, 1][0]
                lr_prob = lr_model.predict_proba(X)[:, 1][0] if lr_model else None

                nn_prob = None
                if nn_model:
                    try:
                        nn_prob = nn_model.predict(X).ravel()[0]
                    except:
                        nn_prob = None

                # Store in session state
                st.session_state['prediction_result'] = {
                    'rf_prob': float(rf_prob),
                    'lr_prob': float(lr_prob) if lr_prob else None,
                    'nn_prob': float(nn_prob) if nn_prob else None,
                }
                st.session_state['prediction_failed'] = False

                st.subheader("📌 Model Predictions:")
                st.write(f"**Random Forest Probability:** {rf_prob:.2f}")
                if lr_prob is not None:
                    st.write(f"**Logistic Regression Probability:** {lr_prob:.2f}")

                if nn_prob is not None:
                    st.write(f"**Neural Network Probability:** {nn_prob:.2f}")

                st.subheader("🧠 Final Decision (Random Forest):")
                if rf_prob > 0.5:
                    st.error("⚠️ High risk of heart disease")
                else:
                    st.success("✅ Low risk of heart disease")
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.session_state['prediction_failed'] = True
                st.session_state['prediction_result'] = None

# Show decision button - persistent across reruns
st.markdown("---")
if st.button("Show Decision (0/1)"):
    if st.session_state['prediction_failed'] or st.session_state['prediction_result'] is None:
        st.warning("Decision not available because prediction failed.")
    else:
        pred_result = st.session_state['prediction_result']
        rf_prob = pred_result.get('rf_prob')
        label = 1 if rf_prob > 0.5 else 0
        st.write(f"**Predicted label (0 = No disease, 1 = Disease):** {label}")
