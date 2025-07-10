AVIAMMO – BIOFUEL INTELLIGENCE SYSTEM
This project focuses on developing an intelligent system that monitors ammonia toxicity levels in poultry environments and predicts future values using machine learning. The system collects real-time data from environmental sensors, performs predictive analysis using models like Random Forest and Fuzzy Logic, and raises alerts when thresholds are breached. It also promotes sustainable biofuel conversion from poultry litter, making it a health-aware and eco-friendly solution.

TABLE OF CONTENTS:

Features

Tech Stack

Usage

System Design

Results

FEATURES:

Real-time ammonia level detection using gas, temperature, and humidity sensor inputs

Predictive modeling using Random Forest Regressor and Fuzzy Logic

Alert system based on toxicity thresholds (visual + console-based)

Interactive data visualization of sensor inputs and predicted trends

Conversion estimation logic for poultry litter to biofuel

Dashboard-ready structure with extendable architecture

TECH STACK:
Language: Python 3.x
Libraries: scikit-learn, NumPy, pandas, matplotlib, seaborn, fuzzy-c-means, time, smtplib (optional for alerts)
Hardware layer (simulated or real): Gas sensor, Temperature sensor, Humidity sensor

USAGE:
Run the script to simulate real-time sensor readings, detect dangerous ammonia levels, and predict future concentrations using machine learning.
The system will also display graphs of recent trends, generate alerts when conditions are unsafe, and output approximate energy recovery estimations if poultry litter is processed into biofuel.

Use the simulate_sensor_data() and predict_ammonia_levels() functions to test system stability, and modify threshold values in the code to match real-world scenarios.

SYSTEM DESIGN:

DATA COLLECTION:

Inputs:

Temperature (°C)

Humidity (%)

Gas concentration (ppm)

PROCESSING MODULES:

Feature scaling and smoothing

Fuzzy logic layer to interpret toxicity conditions

Random Forest model to predict next-hour ammonia levels

Alert module that logs high-risk events and suggests action

VISUALIZATION:

Line plots for sensor input over time

Predicted vs. actual ammonia concentration comparison

Risk-level heatmap (optional)

BIOFUEL ESTIMATION LOGIC:

Poultry litter quantity input

Prediction of energy yield based on moisture and ammonia level

Output of usable energy and emission savings

RESULTS:

The system achieved over 93% R² score using Random Forest for next-hour ammonia prediction

Fuzzy logic module identified risk levels with high reliability based on sensor thresholds

Real-time visualization showed clear rising patterns in gas concentration under unsafe conditions

The system successfully flagged high ammonia exposure (>25 ppm) with immediate alerts

Biofuel module estimated energy outputs and helped link pollution control with energy sustainability

Ideal for use in smart poultry farming or agricultural IoT systems
