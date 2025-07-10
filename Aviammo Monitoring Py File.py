import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz
import datetime
import time
import warnings
import json
import os
warnings.filterwarnings('ignore')


SAVE_DIR = "output"
os.makedirs(SAVE_DIR, exist_ok=True)


def generate_sensor_data(n=1000):
    timestamps = pd.date_range(start="2024-01-01", periods=n, freq='H')
    temperature = np.random.normal(30, 3, n)  # degrees Celsius
    humidity = np.random.normal(60, 10, n)    # %
    gas_level = np.random.normal(5, 1.5, n)   # ppm (raw ammonia sensor value)

    ammonia_ppm = 0.03 * temperature + 0.02 * humidity + 0.5 * gas_level + np.random.normal(0, 0.5, n)
    ammonia_ppm = np.clip(ammonia_ppm, 0, None)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperature,
        'humidity': humidity,
        'gas_level': gas_level,
        'ammonia_ppm': ammonia_ppm
    })
    return df


def fuzzy_ammonia_alert(temp, hum, gas):
    temp_norm = fuzz.interp_membership([0, 50], fuzz.trimf([15, 30, 45]), temp)
    hum_norm = fuzz.interp_membership([0, 100], fuzz.trimf([30, 60, 90]), hum)
    gas_norm = fuzz.interp_membership([0, 10], fuzz.trimf([2, 5, 8]), gas)

    danger_score = (0.3 * temp_norm + 0.3 * hum_norm + 0.4 * gas_norm)

    if danger_score > 0.75:
        return "HIGH RISK – Immediate action needed"
    elif danger_score > 0.5:
        return "MODERATE RISK – Monitor closely"
    else:
        return "SAFE – Within normal limits"


def train_predictive_model(df):
    X = df[['temperature', 'humidity', 'gas_level']]
    y = df['ammonia_ppm']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n[Model Evaluation]")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    pd.DataFrame({
        'True': y_test,
        'Predicted': y_pred
    }).reset_index(drop=True).to_csv(os.path.join(SAVE_DIR, "model_predictions.csv"), index=False)

    return model


def plot_sensor_trends(df):
    plt.figure(figsize=(14, 6))
    plt.plot(df['timestamp'], df['ammonia_ppm'], label='Ammonia Level (ppm)', color='crimson')
    plt.title("Ammonia Levels Over Time")
    plt.xlabel("Time")
    plt.ylabel("Ammonia (ppm)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ammonia_trend.png"))
    plt.show()

    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation between Sensor Readings")
    plt.savefig(os.path.join(SAVE_DIR, "correlation_heatmap.png"))
    plt.show()


def log_sensor_data(logs):
    path = os.path.join(SAVE_DIR, "realtime_logs.json")
    with open(path, 'w') as f:
        json.dump(logs, f, indent=4)


def simulate_real_time_monitoring(model, duration=10):
    print("\n[Real-Time Monitoring Simulation]")
    logs = []
    for i in range(duration):
        temp = random.uniform(25, 35)
        hum = random.uniform(50, 80)
        gas = random.uniform(4, 7)

        ammonia_pred = model.predict([[temp, hum, gas]])[0]
        status = fuzzy_ammonia_alert(temp, hum, gas)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        entry = {
            "time": timestamp,
            "temperature": round(temp, 2),
            "humidity": round(hum, 2),
            "gas_level": round(gas, 2),
            "predicted_ammonia": round(ammonia_pred, 2),
            "alert": status
        }
        logs.append(entry)

        print(f"[Time: {timestamp}]")
        print(f"Temp: {entry['temperature']}°C, Humidity: {entry['humidity']}%, Gas: {entry['gas_level']} ppm")
        print(f"Predicted Ammonia: {entry['predicted_ammonia']} ppm")
        print(f"Status: {status}\n")
        time.sleep(0.5)

    log_sensor_data(logs)


def export_model_summary(model, filename='model_summary.txt'):
    path = os.path.join(SAVE_DIR, filename)
    with open(path, 'w') as f:
        f.write(str(model))


if __name__ == '__main__':
    print("[INFO] Generating synthetic sensor data...")
    df = generate_sensor_data(1200)
    plot_sensor_trends(df)

    print("[INFO] Training ML model for ammonia prediction...")
    model = train_predictive_model(df)

    export_model_summary(model)

    simulate_real_time_monitoring(model, duration=15)
    print("[INFO] AviAmmo monitoring simulation complete. Logs and plots saved in ./output/")

