import numpy as np
import joblib
from keras.models import load_model
import warnings

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

def predict(input_values):
    """Validate inputs and predict"""
    if len(input_values) != 16:
        raise ValueError(f"Need 16 features, got {len(input_values)}")
    
    try:
        # Load artifacts
        model = load_model('./models/traffic_model.h5')
        scaler = joblib.load('./models/scaler.pkl')
        
        # Preprocess
        input_array = np.array(input_values).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        
        # Predict
        return model.predict(scaled_input.reshape(1, 1, -1))[0][0]
        
    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    print(predict([0]*16))  # Test with dummy input
