import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def load_data(traffic_file, weather_file):
    traffic = pd.read_csv(traffic_file)
    weather = pd.read_csv(weather_file, parse_dates=['Date.Full'])
    weather.rename(columns={'Date.Full': 'timestamp'}, inplace=True)
    return traffic, weather

def preprocess_data(traffic, weather):
    # Merge datasets
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    traffic['timestamp'] = pd.date_range(
        start=weather['timestamp'].min(),
        end=weather['timestamp'].max(),
        periods=len(traffic)
    )
    merged = pd.merge(traffic, weather, on='timestamp')
    
    # Feature engineering
    merged['hour'] = merged['timestamp'].dt.hour
    merged['day_of_week'] = merged['timestamp'].dt.dayofweek
    merged['month'] = merged['timestamp'].dt.month
    merged['is_weekend'] = merged['timestamp'].dt.dayofweek // 5

    # Define features
    FEATURE_COLUMNS = [
        'link_length_km',
        'pedal_cycles',
        'two_wheeled_motor_vehicles',
        'cars_and_taxis',
        'buses_and_coaches',
        'LGVs',
        'all_HGVs',
        'Data.Precipitation',
        'Data.Temperature.Avg Temp',
        'Data.Temperature.Max Temp',
        'Data.Temperature.Min Temp',
        'Data.Wind.Speed',
        'hour',
        'day_of_week',
        'month',
        'is_weekend'
    ]
    
    # Target
    merged['target'] = merged['all_motor_vehicles'].shift(-1).ffill()
    
    # Filter and preprocess
    data = merged[FEATURE_COLUMNS + ['target']].copy()
    data.ffill(inplace=True)
    
    # Scaling
    scaler = MinMaxScaler()
    data[FEATURE_COLUMNS] = scaler.fit_transform(data[FEATURE_COLUMNS])
    
    # Save artifacts
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, './models/scaler.pkl')
    joblib.dump(FEATURE_COLUMNS, 'models/feature_columns.pkl')
    
    data.to_csv('./data/processed_data.csv', index=False)
    return data

if __name__ == '__main__':
    traffic, weather = load_data('./data/traffic_data.csv', './data/weather_data.csv')
    preprocess_data(traffic, weather)
