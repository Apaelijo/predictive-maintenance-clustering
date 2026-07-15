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


**Technical Documentation: How To Use This Project**

This project delivers a predictive maintenance dashboard that groups machine behavior into operating regimes using clustering and then translates those results into practical maintenance guidance. It supports both technical users (data preparation, model evaluation, explainability, fairness checks) and non-technical users (simple dashboard interaction, risk labels, and action-oriented status outputs).

**1. Project Purpose**
- Detect machine operating patterns from telemetry data
- Highlight abnormal or higher-risk behavior early
- Support proactive maintenance decisions through an interactive dashboard
- Provide model evaluation, explainability, and fairness analysis notebooks

**2. Main Components**
- Dashboard application: app.py
- Data loading and cleaning: preprocess.py
- Feature engineering and preprocessing pipeline: build_features.py
- Clustering utilities: cluster.py
- Training script (model experiments): train.py
- Evaluation notebook (includes fairness audit): 03_evaluation.ipynb
- Dependency list: requirements.txt

**3. Prerequisites**
- Windows, macOS, or Linux
- Python 3.8 or newer
- Internet access for package installation
- Dataset present at ai4i2020.csv

**4. Installation and Environment Setup**
1. Open a terminal in the project root folder.
2. Create a virtual environment:
   python -m venv .venv
3. Activate it on Windows:
   .venv\Scripts\activate
4. Install dependencies:
   pip install -r requirements.txt

**5. Standard Usage Flow**
1. Data understanding and preprocessing
- Run 01_eda_preprocessing.ipynb to inspect data quality, distributions, and prepared features.

2. Clustering development
- Run 02_clustering.ipynb to build and compare clustering approaches and select the champion model.

3. Evaluation and governance checks
- Run 03_evaluation.ipynb to review:
- Cluster quality and distribution
- Failure-rate analysis by cluster
- Explainability summary
- Bias and fairness checks by machine Type (L, M, H)

4. Save deployment artifacts
- Ensure the following files exist before running the dashboard:
- preprocessing_pipeline.pkl
- champion_dbscan.pkl
- Optional for plotting historical cluster labels:
- cluster_assignments.csv

5. Launch the dashboard
- Start Streamlit:
  streamlit run app.py
- Open the local URL shown in the terminal (typically localhost on port 8501).

**6. How End Users Operate the Dashboard**
- Enter machine telemetry from the sidebar (Type, temperatures, speed, torque, tool wear), then apply inputs
- Optionally enable Live Sensor Stream for continuous simulated updates
- Read the detected operating condition and risk badge
- Use the telemetry cards and scatter chart to compare current behavior with historical operating regions
- Open the guidance tab for plain-language interpretation and recommended maintenance actions

**7. Expected Inputs and Outputs**
- Inputs:
- Machine Type (L, M, H)
- Air temperature, process temperature, rotational speed, torque, tool wear
- Outputs:
- Cluster assignment
- Human-readable operating condition label
- Risk color/status and maintenance guidance
- Visual position of current point against historical speed-torque behavior

**8. Important Implementation Notes**
- The app is currently wired for DBSCAN deployment artifacts in app.py.
- The experimental training script train.py includes K-Means champion saving logic by default.
- If you use the training script directly, align saved artifact naming and model selection with the dashboard expectations, or update dashboard model paths accordingly.

**9. Troubleshooting**
- App starts but predictions look random:
- Check whether real model files are missing; fallback mode may be active.
- Model loading errors:
- Confirm artifact paths and file names in app.py match files in the models folder.
- Data file not found:
- Verify ai4i2020.csv exists.
- Notebook import issues:
- Run from project root so module paths resolve correctly.
- Streamlit dependency missing:
- Reinstall dependencies from requirements.txt.


### Using cluster_assignments.csv for Validation

The file cluster_assignments.csv is used as a reference label source for historical visualization and validation checks. In the app logic, app.py loads this file when model labels are unavailable and expects a column named dbscan_cluster with the same number of rows as the historical dataset in ai4i2020.csv, as shown in app.py.

- Purpose:
  - Preserve stable, reproducible cluster labels for evaluation and dashboard consistency
  - Compare newly generated cluster assignments against a known baseline
  - Support governance checks in 03_evaluation.ipynb

- Minimum file requirements:
  - Must exist at cluster_assignments.csv
  - Must include column dbscan_cluster
  - Row count must match ai4i2020.csv

- Recommended validation workflow:
  1. Regenerate clusters from the latest pipeline and model
  2. Join or align regenerated labels with the existing assignment file by row order (or by a retained key if available)
  3. Check agreement rate between old and new assignments
  4. Review failure-rate-by-cluster shifts before replacing the baseline
  5. Approve and version the updated assignment file only if drift is expected and justified

- Quick quality checks to record in release notes:
  - Row-count parity between raw data and assignment file
  - Presence and datatype of dbscan_cluster
  - Percentage of noise points where cluster equals -1
  - Failure-rate trend by cluster and by Type group (L, M, H)

