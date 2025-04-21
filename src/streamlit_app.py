# Add environment variables FIRST
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import streamlit as st
from model_inference import predict
import joblib
from keras.models import load_model

# Render UI
st.set_page_config(layout="wide")
st.title("🚦 Traffic Volume Predictor")
st.write("Enter values and click Predict")

# Load artifacts with improved error handling
MODEL_FILES = {
    'feature_columns': 'models/feature_columns.pkl',
    'scaler': 'models/scaler.pkl',
    'model': 'models/traffic_model.h5'
}

try:
    FEATURE_COLUMNS = joblib.load(MODEL_FILES['feature_columns'])
    scaler = joblib.load(MODEL_FILES['scaler'])
    model = load_model(MODEL_FILES['model'])
    st.success("✅ Models loaded successfully!")
except FileNotFoundError as e:
    st.error(f"❌ Missing model file: {e.filename}")
    st.stop()
except Exception as e:
    st.error(f"❌ Model loading failed: {str(e)}")
    st.stop()

# Input fields (unchanged but kept for completeness)
inputs = []
col1, col2 = st.columns(2)

with col1:
    inputs.append(st.number_input("Link Length (km)", min_value=0.0, value=48.75))
    inputs.append(st.number_input("Pedal Cycles", min_value=0, value=228))
    inputs.append(st.number_input("Motorcycles", min_value=0, value=289))
    inputs.append(st.number_input("Cars/Taxis", min_value=0, value=19674))
    inputs.append(st.number_input("Buses", min_value=0, value=245))
    inputs.append(st.number_input("LGVs", min_value=0, value=2351))
    inputs.append(st.number_input("HGVs", min_value=0, value=1295))
    inputs.append(st.number_input("Precipitation (mm)", min_value=0.0, value=0.0))

with col2:
    inputs.append(st.number_input("Avg Temp (°C)", value=15.0))
    inputs.append(st.number_input("Max Temp (°C)", value=20.0))
    inputs.append(st.number_input("Min Temp (°C)", value=10.0))
    inputs.append(st.number_input("Wind Speed (km/h)", min_value=0.0, value=4.33))
    inputs.append(st.number_input("Hour of Day", min_value=0, max_value=23, value=12))
    inputs.append(st.number_input("Day of Week", min_value=0, max_value=6, value=3))
    inputs.append(st.number_input("Month", min_value=1, max_value=12, value=3))
    inputs.append(st.number_input("Is Weekend?", min_value=0, max_value=1, value=0))

if st.button("Predict Traffic", type="primary"):
    try:
        # Validate inputs
        inputs = [float(x) for x in inputs]
        if len(inputs) != len(FEATURE_COLUMNS):  # Dynamic check
            st.error(f"⚠️ Requires {len(FEATURE_COLUMNS)} features, got {len(inputs)}")
            st.stop()
            
        # Predict
        pred = predict(inputs)
        st.success(f"**Predicted Traffic Volume:** {pred:,.2f} vehicles")
        
    except ValueError as e:
        st.error(f"❌ Invalid input: {str(e)}")
    except Exception as e:
        st.error(f"🚨 Prediction failed: {str(e)}")

# Add debug info (optional)
with st.expander("Debug Info"):
    st.write(f"Features expected: {FEATURE_COLUMNS}")
    st.write(f"Last prediction: {pred if 'pred' in locals() else 'None'}")
