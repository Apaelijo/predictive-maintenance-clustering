# Predictive Maintenance Clustering System (DBSCAN + SHAP)

[![License](https://img.shields.io/badge/License-Placeholder-lightgrey.svg)](#license)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

## Overview

The Predictive Maintenance Clustering System is a machine telemetry dashboard for identifying operating regimes, monitoring current machine state, and supporting proactive maintenance decisions. The application uses DBSCAN clustering to group historical machine behavior into meaningful regimes such as abnormal load, variable operation, and stable running.

The project also includes SHAP explainability to help interpret which input signals contribute most strongly to the model’s behavior and maintenance interpretation. In the dashboard, operators can monitor the current machine state in real time while comparing it against historical speed-torque patterns and cluster assignments.

## Repository Structure

```text
predictive-maintenance-clustering/
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   └── processed/
│       └── cluster_assignments.csv
├── models/
│   ├── champion_dbscan.pkl
│   └── preprocessing_pipeline.pkl
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_clustering.ipynb
│   ├── 03_evaluation.ipynb
│   └── Christopher_Lopez_Capstone_Technical_Presentation.ipynb
├── presentation/
│   ├── images/
│   ├── Christopher_Lopez_Capstone_Business_Facing_Presentation.pptx
│   └── Christopher_Lopez_Capstone_Technical_Document.docx
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── data/
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── cluster.py
│   │   └── train.py
│   └── utils/
│       └── viz.py
├── requirements.txt
├── README.md
└── LICENSE