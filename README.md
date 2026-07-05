# Predictive Maintenance Clustering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An open-source data science project focused on identifying and classifying distinct machine failure modes using unsupervised machine learning. By clustering operational telemetric data, this pipeline helps maintenance teams transition from reactive fixes to targeted, proactive root-cause interventions.

---

## 📌 Business Case & Overview
In modern manufacturing, unexpected equipment downtime is incredibly costly. While traditional predictive maintenance focuses on binary classification (will it fail or not?), this project leverages **unsupervised clustering** to uncover the *hidden signatures* of different root-cause failure modes (e.g., Tool Wear, Heat Dissipation, Power Failure, and Overstrain). 

By automatically segmenting these anomalies, operations teams can deploy specific engineering fixes rather than wasting time diagnosing the issue from scratch.

## 📊 Dataset
This project utilizes the **AI4I 2020 Predictive Maintenance Dataset**, a synthetic dataset reflecting real-world industrial milling machine operations. 
* **Features used:** Air temperature, process temperature, rotational speed, torque, and tool wear.

---

## 📂 Project Structure


predictive-maintenance-clustering/
├── data/
│   ├── raw/            # Immutable, original data source (e.g., ai4i2020.csv)
│   └── processed/      # Cleaned, transformed, and scaled data for modeling
├── notebooks/
│   └── 01_eda_preprocessing.ipynb  # Interactive data exploration & visualization
├── src/                # Production-ready, modular Python code
│   ├── __init__.py
│   ├── data/
│   │   └── preprocess.py          # Data ingestion and cleaning pipelines
│   ├── features/
│   │   └── build_features.py      # Feature scaling and engineering scripts
│   └── utils/
│       └── viz.py                 # Reusable plotting functions for clustering
├── models/             # Saved model weights/artifacts (e.g., GMM pickle files)
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Files protected from git tracking


About the Sample Application

Core Features Delivered ✅
1. Pipeline & Model Loading
✅ Cached loading of preprocessing_pipeline.pkl and champion_kmeans.pkl using @st.cache_resource
✅ Graceful fallback to mock models if files aren't found yet
✅ Comprehensive error handling with logging
2. Interactive UI Sidebar
✅ Type dropdown (L, M, H options)
✅ Air Temperature slider (295-305K)
✅ Process Temperature slider (304-315K)
✅ Rotational Speed slider (1100-2900 rpm)
✅ Torque slider (3.0-75.0 Nm)
✅ Tool Wear slider (0-250 min)
3. Feature Engineering Engine
✅ Temp_Diff = Process Temperature - Air Temperature
✅ Power_Proxy = Torque × Rotational Speed
Dynamically calculated from user inputs
4. Operational Status Display
✅ Prominent cluster prediction card
✅ Risk mapping system:
🟢 Stable Operators (Cluster 0) - Green/Success
ℹ️ Efficient but Young Machines (Cluster 1) - Blue/Info
🟠 High-Load Machines (Cluster 2) - Orange/Warning
🔴 Aging or At-Risk Machines (Cluster 3) - Red/Error
5. Interactive Plotly Visualizations
✅ 2D scatter plot: Rotational Speed vs Torque
✅ Historical background points (lightly colored)
✅ Large, prominent red star marker for current machine state
✅ Interactive hover information
6. Automated Streamer Mode
✅ Sidebar toggle: "Enable Live Sensor Stream"
✅ Adjustable stream speed (0.5, 1.0, 2.0, 5.0 seconds)
✅ Realistic fluctuating sensor readings
✅ Live event logging (last 10 updates)
✅ Real-time dashboard updates
7. Additional Features
✅ Clean, production-ready code with extensive comments
✅ Modern Streamlit layout with st.columns() and st.metric()
✅ Comprehensive error handling throughout
✅ Professional styling and responsive design
✅ Emoji-enhanced UI for better readability
✅ Full logging configuration
Key Implementation Details
Caching Strategy: Models loaded once and reused across sessions for optimal performance
Fallback Mechanism: Application works even if .pkl files haven't been created yet
Session State Management: Maintains last sensor reading for realistic fluctuations during streaming
Feature-Complete: All engineered features calculated on-the-fly from raw inputs
Real Dataset Alignment: Uses exact feature names and ranges from AI4I 2020 dataset