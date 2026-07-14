"""
Production-Ready Streamlit Web Application for Machine Predictive Maintenance Clustering.

This application provides real-time machine telemetry simulation and monitoring with:
- Interactive sidebar controls for manual telemetry input
- Automated live sensor streaming mode
- Feature engineering pipeline
- K-Means clustering predictions with risk assessment
- Interactive Plotly visualizations
- Graceful error handling and model loading with caching

Author: Predictive Maintenance Team
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from pathlib import Path
from typing import Tuple, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import DBSCAN
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Define project root for relative imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Model paths
SCALER_PATH = PROJECT_ROOT / "models" / "preprocessing_pipeline.pkl"
CLUSTER_MODEL_PATH = PROJECT_ROOT / "models" / "champion_dbscan.pkl"

# AI4I 2020 Dataset feature ranges (for realistic data generation)
TELEMETRY_RANGES = {
    'Air temperature [K]': (295.0, 305.0),
    'Process temperature [K]': (304.0, 315.0),
    'Rotational speed [rpm]': (1100, 2900),
    'Torque [Nm]': (3.0, 75.0),
    'Tool wear [min]': (0, 250),
}

TYPE_OPTIONS = ['L', 'M', 'H']

# Risk mapping: cluster_id -> (status_label, color, severity)
RISK_MAPPING = {
    -1: ("Abnormal Load", "warning", 2),
    0: ("Variable Operation", "success", 0),
    1: ("Stable Running", "info", 0),
}

# Extend this mapping if the model is retrained with extra clusters.
# Cluster IDs beyond the known set will still fall back to a generic label.
CLUSTER_VISUALS = {
    -1: {"name": "Abnormal Load", "color": "#E74C3C"},
    0: {"name": "Variable Operation", "color": "#F39C12"},
    1: {"name": "Stable Running", "color": "#3498DB"},
}
# ============================================================================
# MODEL LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_preprocessing_pipeline():
    """
    Load the preprocessing pipeline (scaler + encoder) with caching.
    
    Returns:
        Loaded preprocessing pipeline or None if not found.
        Also returns fallback mock transformer if files unavailable.
    """
    try:
        if not SCALER_PATH.exists():
            logger.warning(f"Preprocessing pipeline not found at {SCALER_PATH}")
            logger.info("Using fallback mock pipeline...")
            return _create_fallback_pipeline()
        
        pipeline = joblib.load(SCALER_PATH)
        logger.info("✓ Preprocessing pipeline loaded successfully")
        return pipeline
    
    except Exception as e:
        logger.error(f"Error loading preprocessing pipeline: {e}")
        logger.info("Falling back to mock pipeline...")
        return _create_fallback_pipeline()


@st.cache_resource
def load_cluster_model():
    """
    Load the trained clustering model with caching.
    
    Returns:
        Loaded clustering model or fallback mock model if not found.
    """
    try:
        if not CLUSTER_MODEL_PATH.exists():
            logger.warning(f"Clustering model not found at {CLUSTER_MODEL_PATH}")
            logger.info("Using fallback mock model...")
            return _create_fallback_cluster_model()
        
        model = joblib.load(CLUSTER_MODEL_PATH)
        logger.info("✓ Clustering model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error loading clustering model: {e}")
        logger.info("Falling back to mock model...")
        return _create_fallback_cluster_model()


def _create_fallback_pipeline():
    """
    Create a minimal fallback pipeline for demo purposes when actual models are unavailable.
    
    Returns:
        Simple mock transformer that passes data through.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    
    numeric_features = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
        ],
        remainder='passthrough'
    )
    
    logger.info("Fallback pipeline created")
    return preprocessor


def _create_fallback_cluster_model():
    """
    Create a minimal fallback clustering model for demo purposes.
    
    Returns:
        Mock clustering object that returns random cluster assignments.
    """
    class MockClusterModel:
        """Mock clustering model for fallback when actual model is unavailable."""
        def __init__(self):
            self.n_clusters = 3
        
        def predict(self, X):
            """Return random cluster assignment (0, 1, or 2)."""
            return np.random.randint(0, self.n_clusters, size=len(X))

        def fit_predict(self, X):
            """Return random cluster assignment (0, 1, or 2) for compatibility with DBSCAN-like models."""
            return self.predict(X)
    
    logger.info("Fallback clustering model created")
    return MockClusterModel()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def engineer_features(sensor_data: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate engineered features from raw telemetry data.
    
    Args:
        sensor_data: Dictionary containing raw telemetry readings
        
    Returns:
        Dictionary with original + engineered features
    """
    engineered = sensor_data.copy()
    
    # Calculate Temp_Diff: Process Temperature - Air Temperature
    engineered['Temp_Diff'] = (
        sensor_data['Process temperature [K]'] - 
        sensor_data['Air temperature [K]']
    )
    
    # Calculate Power_Proxy: Torque * Rotational Speed
    engineered['Power_Proxy'] = (
        sensor_data['Torque [Nm]'] * 
        sensor_data['Rotational speed [rpm]']
    )
    
    return engineered


def prepare_prediction_input(sensor_data: Dict[str, float]) -> pd.DataFrame:
    """
    Prepare raw sensor data for model prediction.
    Handles feature scaling and One-Hot encoding of categorical variables.
    
    Args:
        sensor_data: Raw telemetry readings
        
    Returns:
        Properly formatted DataFrame ready for clustering model
    """
    # Create DataFrame with original features
    df = pd.DataFrame([sensor_data])
    
    # Ensure correct column names and order
    required_cols = [
        'Air temperature [K]', 'Process temperature [K]',
        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type'
    ]
    
    # Reorder columns to match training schema
    df = df[required_cols]
    
    return df


# ============================================================================
# PREDICTION & RISK ASSESSMENT
# ============================================================================

def _is_dbscan_model(model) -> bool:
    """Detect whether the loaded model is a trained DBSCAN clustering object."""
    return (
        hasattr(model, 'components_') and
        hasattr(model, 'core_sample_indices_') and
        hasattr(model, 'labels_') and
        hasattr(model, 'eps')
    )


def _dbscan_predict(model, X, return_debug: bool = False):
    """Assign cluster labels to new points using a fitted DBSCAN model.

    Strategy:
    1) If a point has core neighbors within eps, assign by local majority vote.
    2) If not, use nearest-core fallback only when reasonably close to eps.
    """
    from sklearn.neighbors import NearestNeighbors

    if not _is_dbscan_model(model):
        raise AttributeError("DBSCAN model missing required trained attributes")

    core_samples = model.components_
    core_labels = model.labels_[model.core_sample_indices_]
    if core_samples.size == 0:
        preds = np.full(len(X), -1, dtype=int)
        if return_debug:
            return preds, [{'reason': 'no_core_samples'} for _ in range(len(X))]
        return preds

    nn_radius = NearestNeighbors(radius=model.eps).fit(core_samples)
    distances, neighbors = nn_radius.radius_neighbors(X)
    k = min(5, len(core_samples))
    nn_knn = NearestNeighbors(n_neighbors=k).fit(core_samples)

    fallback_multiplier = 1.35
    preds = []
    debug_rows = []
    for dist_list, neighbor_idx in zip(distances, neighbors):
        if len(neighbor_idx) == 0:
            nearest_dist, nearest_idx = nn_knn.kneighbors([X[len(preds)]], n_neighbors=1)
            nearest_distance = float(nearest_dist[0][0])
            nearest_label = int(core_labels[nearest_idx[0][0]])

            if nearest_distance <= float(model.eps) * fallback_multiplier:
                preds.append(nearest_label)
                debug_rows.append({
                    'assigned_by': 'nearest_core_fallback',
                    'neighbor_count': 0,
                    'nearest_distance': nearest_distance,
                })
            else:
                preds.append(-1)
                debug_rows.append({
                    'assigned_by': 'noise',
                    'neighbor_count': 0,
                    'nearest_distance': nearest_distance,
                })
        else:
            local_labels = core_labels[neighbor_idx]
            unique_labels, counts = np.unique(local_labels, return_counts=True)
            voted_label = int(unique_labels[np.argmax(counts)])
            preds.append(voted_label)
            debug_rows.append({
                'assigned_by': 'radius_majority_vote',
                'neighbor_count': int(len(neighbor_idx)),
                'nearest_distance': float(np.min(dist_list)),
            })

    preds = np.array(preds, dtype=int)
    if return_debug:
        return preds, debug_rows
    return preds


def predict_cluster(
    sensor_data: Dict[str, float],
    pipeline,
    model,
    debug: bool = False
):
    """
    Predict cluster assignment and map to risk status.
    
    Args:
        sensor_data: Raw telemetry readings
        pipeline: Preprocessing pipeline
        model: Trained clustering model
        debug: Whether to include prediction debug details
        
    Returns:
        Tuple of (cluster_id, risk_label, risk_color) or
        (cluster_id, risk_label, risk_color, debug_info) when debug=True
    """
    debug_info = {}
    try:
        # Prepare input data
        df = prepare_prediction_input(sensor_data)

        # Prefer using transform to avoid refitting scaler/pipeline on a single sample.
        # Fall back to fit_transform if the loaded pipeline is unfitted (e.g., a fallback preprocessor).
        try:
            X_processed = pipeline.transform(df)
        except Exception:
            X_processed = pipeline.fit_transform(df)

        debug_info['input_shape'] = X_processed.shape
        debug_info['input_values'] = X_processed.tolist()

        # Use DBSCAN assignment if the loaded model appears to be a fitted DBSCAN.
        if _is_dbscan_model(model):
            if debug:
                preds, dbscan_debug = _dbscan_predict(model, X_processed, return_debug=True)
                debug_info['dbscan_details'] = dbscan_debug
            else:
                preds = _dbscan_predict(model, X_processed)
            debug_info['prediction_method'] = 'dbscan_radius_assignment'
        elif hasattr(model, 'predict'):
            preds = model.predict(X_processed)
            debug_info['prediction_method'] = 'predict'
        elif hasattr(model, 'fit_predict'):
            preds = model.fit_predict(X_processed)
            debug_info['prediction_method'] = 'fit_predict'
        else:
            raise AttributeError("Loaded model has neither predict nor fit_predict")

        cluster_id = int(preds[0]) if len(preds) > 0 else -1
        debug_info['cluster_id'] = cluster_id

        # Map to risk status
        risk_label, risk_color, _ = RISK_MAPPING.get(
            cluster_id,
            ("Unknown Cluster", "secondary", -1)
        )

        if debug:
            return cluster_id, risk_label, risk_color, debug_info
        return cluster_id, risk_label, risk_color

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        debug_info['error'] = str(e)
        if debug:
            return -1, "Prediction Error", "secondary", debug_info
        return -1, "Prediction Error", "secondary"


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_sidebar_telemetry_controls() -> Dict[str, float]:
    """
    Create interactive sidebar controls for manual telemetry input.
    
    Returns:
        Dictionary of user-selected sensor readings
    """
    with st.sidebar.expander("⚙️ Machine Telemetry Input Panel", expanded=False):
        st.markdown("### 📊 Sensor Readings")

        def _parse_numeric_input(value: str, default: float, cast_type):
            try:
                return cast_type(value)
            except (TypeError, ValueError):
                return default

        # Type dropdown
        machine_type = st.selectbox(
            "🏭 Machine Type",
            options=TYPE_OPTIONS,
            help="Machine profile: Low (L), Medium (M), or High (H) stress configuration"
        )

        # Air Temperature text input
        air_temp = _parse_numeric_input(
            st.text_input(
                "🌡️ Air Temperature [K]",
                value="300.0",
                help="Ambient air temperature around the machine"
            ),
            300.0,
            float,
        )

        # Process Temperature text input
        process_temp = _parse_numeric_input(
            st.text_input(
                "🔥 Process Temperature [K]",
                value="310.0",
                help="Core operating temperature of the machine"
            ),
            310.0,
            float,
        )

        # Rotational Speed text input
        rot_speed = _parse_numeric_input(
            st.text_input(
                "⚡ Rotational Speed [rpm]",
                value="1500",
                help="Spindle/motor rotation speed"
            ),
            1500,
            int,
        )

        # Torque text input
        torque = _parse_numeric_input(
            st.text_input(
                "🔧 Torque [Nm]",
                value="40.0",
                help="Rotational force applied by the spindle"
            ),
            40.0,
            float,
        )

        # Tool Wear text input
        tool_wear = _parse_numeric_input(
            st.text_input(
                "⏱️ Tool Wear [min]",
                value="100",
                help="Cumulative tool wear in minutes of operation"
            ),
            100,
            int,
        )

        deploy_clicked = st.button("Apply Inputs", use_container_width=True, key="deploy_panel")
        if deploy_clicked:
            st.session_state["deploy_requested"] = True
            st.toast("Current telemetry inputs applied")
    
    # Assemble sensor dictionary
    sensor_data = {
        'Type': machine_type,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': process_temp,
        'Rotational speed [rpm]': rot_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
    }
    
    return sensor_data


def render_metric_card(icon: str, title: str, value: str, detail: Optional[str] = None):
    """Render a compact metric tile with icon, title, and value on one line."""
    detail_html = f"<div style='font-size:0.78rem; color:#6b7280; margin-top:4px;'>{detail}</div>" if detail else ""
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb; border-radius:10px; padding:10px 12px; background:#f8fafc; min-height:74px;">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:8px;">
                <div style="font-size:1.1rem;">{icon}</div>
                <div style="flex:1; min-width:0;">
                    <div style="font-size:0.8rem; color:#6b7280;">{title}</div>
                    <div style="font-size:1.02rem; font-weight:600; color:#111827;">{value}</div>
                </div>
            </div>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_operational_status(cluster_id: int, risk_label: str, risk_color: str):
    """
    Display prominent operational status card with risk badge.
    
    Args:
        cluster_id: Predicted cluster assignment
        risk_label: Human-readable risk status
        risk_color: Color coding for status (success, warning, error)
    """
    # Render status area full-width so it uses the entire app pane
    with st.container():
        st.markdown("---")

        # Primary title showing full cluster label (use neutral wording)
        st.markdown(f"## 🎯 Detected Operating Condition\n\n**Cluster {cluster_id}: {risk_label}**")

        # Colored badge (pill) matching the risk_color semantics
        emoji_map = {
            "success": "✅",
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "🚨",
        }

        color_map = {
            "success": ("#dcfce7", "#065f46"),
            "info": ("#dbeafe", "#1e3a8a"),
            "warning": ("#fffbeb", "#92400e"),
            "error": ("#fee2e2", "#7f1d1d"),
        }

        bg, fg = color_map.get(risk_color, ("#f3f4f6", "#374151"))
        emoji = emoji_map.get(risk_color, "ℹ️")

        badge_html = (
            f"<div style=\"display:inline-block; padding:10px 18px; border-radius:10px;"
            f" background:{bg}; color:{fg}; font-weight:600; margin-top:12px;\">{emoji} {risk_label}</div>"
        )

        # Use full-width markdown so the badge and caption span the pane
        st.markdown(badge_html, unsafe_allow_html=True)

        # Concise maintenance guidance (single-line caption under the badge)
        if risk_color == "success":
            st.caption("Maintenance action: Continue routine monitoring and schedule standard inspections.")
        elif risk_color == "info":
            st.caption("Maintenance action: Monitor performance; no immediate action needed.")
        elif risk_color == "warning":
            st.caption("Maintenance action: Schedule preventive checks; risk of wear or overheating.")
        elif risk_color == "error":
            st.caption("Maintenance action: Prioritize inspection; possible early intervention needed.")
        else:
            st.caption("Maintenance action: Validate the input and reassess the operating state.")

        st.markdown("---")


def render_telemetry_metrics(sensor_data: Dict[str, float]):
    """
    Display key telemetry metrics in a compact 3x2 grid layout.
    
    Args:
        sensor_data: Current sensor readings
    """
    st.subheader("📡 Current Machine Health Snapshot")

    metric_items = [
        ("🌡️", "Air Temp [K]", f"{sensor_data['Air temperature [K]']:.1f} K", f"vs. baseline {sensor_data['Air temperature [K]'] - 300:+.1f} K"),
        ("🔥", "Process Temp [K]", f"{sensor_data['Process temperature [K]']:.1f} K", f"vs. baseline {sensor_data['Process temperature [K]'] - 310:+.1f} K"),
        ("⚡", "Rotational Speed", f"{sensor_data['Rotational speed [rpm]']:,.0f} rpm", "Load profile"),
        ("🔧", "Torque", f"{sensor_data['Torque [Nm]']:.1f} Nm", "Mechanical force"),
        ("⏱️", "Tool Wear", f"{sensor_data['Tool wear [min]']:.0f} min", "Accumulated wear"),
    ]

    temp_diff = sensor_data['Process temperature [K]'] - sensor_data['Air temperature [K]']
    metric_items.append(("📊", "Heat Dissipation Proxy [K]", f"{temp_diff:.1f} K", "Engineered feature"))

    for row in range(2):
        cols = st.columns(3)
        for col_idx in range(3):
            item_idx = row * 3 + col_idx
            if item_idx < len(metric_items):
                icon, title, value, detail = metric_items[item_idx]
                with cols[col_idx]:
                    render_metric_card(icon, title, value, detail)


@st.cache_data(show_spinner=False)
def load_historical_speed_torque_data() -> pd.DataFrame:
    """Load historical speed-torque data used for the cluster visualization."""
    raw_data_path = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
    df = pd.read_csv(raw_data_path)

    df = df.drop(columns=["UDI", "Product ID"], errors="ignore")
    df = df.drop(columns=["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"], errors="ignore")

    return df[["Rotational speed [rpm]", "Torque [Nm]"]].copy()


def _resolve_historical_dbscan_labels(cluster_model, n_rows: int) -> np.ndarray:
    """Resolve historical DBSCAN labels from the fitted model or saved assignments."""
    if _is_dbscan_model(cluster_model) and len(getattr(cluster_model, "labels_", [])) == n_rows:
        return np.asarray(cluster_model.labels_)

    assignments_path = PROJECT_ROOT / "data" / "processed" / "cluster_assignments.csv"
    if assignments_path.exists():
        assignments_df = pd.read_csv(assignments_path)
        if "dbscan_cluster" in assignments_df.columns and len(assignments_df) == n_rows:
            return assignments_df["dbscan_cluster"].to_numpy()

    return np.full(n_rows, -1, dtype=int)


def render_interactive_scatter_plot(sensor_data: Dict[str, float], cluster_id: int, cluster_model):
    """
    Create interactive 2D Plotly scatter plot: Rotational Speed vs Torque.
    
    Historical points are colored by DBSCAN cluster label and the current machine state is
    emphasized with a prominent marker.
    
    Args:
        sensor_data: Current sensor readings for the current machine state
        cluster_id: Predicted DBSCAN cluster label for the current machine state
        cluster_model: Loaded DBSCAN model used to color the historical points
    """
    st.subheader("🎯 DBSCAN Operating Regimes in Speed-Torque Space")

    historical_df = load_historical_speed_torque_data()
    historical_labels = _resolve_historical_dbscan_labels(cluster_model, len(historical_df))

    fig = go.Figure()

    for label in [-1, 0, 1]:
        label_mask = historical_labels == label
        if not np.any(label_mask):
            continue

        visual = CLUSTER_VISUALS[label]
        fig.add_trace(go.Scatter(
            x=historical_df.loc[label_mask, "Rotational speed [rpm]"],
            y=historical_df.loc[label_mask, "Torque [Nm]"],
            mode="markers",
            name=visual["name"],
            legendgroup=str(label),
            marker=dict(
                size=7,
                color=visual["color"],
                opacity=0.7,
                line=dict(color="white", width=0.5),
            ),
            hovertemplate=(
                f"<b>{visual['name']}</b><br>"
                "Speed: %{x:.0f} rpm<br>"
                "Torque: %{y:.1f} Nm<extra></extra>"
            ),
        ))

    current_speed = sensor_data['Rotational speed [rpm]']
    current_torque = sensor_data['Torque [Nm]']
    current_visual = CLUSTER_VISUALS.get(cluster_id, {"name": f"Cluster {cluster_id}", "color": "#6C757D"})
    
    fig.add_trace(go.Scatter(
        x=[current_speed],
        y=[current_torque],
        mode="markers+text",
        marker=dict(
            size=22,
            color=current_visual["color"],
            symbol="star",
            line=dict(color="black", width=2.5),
            opacity=1.0,
        ),
        text=["CURRENT"],
        textposition="top center",
        textfont=dict(size=12, color="black"),
        name="Current Machine State",
        showlegend=False,
        hovertemplate=(
            f"<b>Current Machine State</b><br>Cluster: {current_visual['name']}<br>"
            "Speed: %{x:.0f} rpm<br>"
            "Torque: %{y:.1f} Nm<extra></extra>"
        ),
    ))
    
    # Update layout
    fig.update_layout(
        title="Machine Operating Regimes in Speed-Torque Space",
        xaxis_title="Rotational Speed [rpm]",
        yaxis_title="Torque [Nm]",
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=900,
        height=520,
        showlegend=True,
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
        ),
        margin=dict(l=30, r=20, t=70, b=30),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        zeroline=False,
        rangemode="tozero",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        zeroline=False,
        rangemode="tozero",
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_cluster_guidance_tab():
    """Render an information tab explaining the clusters and maintenance actions."""
    st.subheader("🧠 How to Read the Clusters")
    st.markdown("""
    The app groups machine behavior into operating regimes using telemetry patterns.
    These regimes help maintenance teams interpret the current machine condition without waiting for a failure.
    """)

    st.markdown("### Cluster meanings")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.warning("⚠️ Cluster -1 - Abnormal Load")
        st.write("Highest and most variable power proxy, lowest temperature differential, and highest tool wear.")
        st.caption("State: Signs of failure or worth investigating for maintenance.")

    with col2:
        st.success("✅ Cluster 0 - Variable Operation")
        st.write("Medium-high power proxy, medium temperature differential, and medium tool wear.")
        st.caption("State: Represents typical machine behavior; operating normally with some variability.")

    with col3:
        st.info("ℹ️ Cluster 1 - Stable Running")
        st.write("Lowest power proxy, highest temperature differential, and lowest tool wear.")
        st.caption("State: Stable, low-stress operating regime; optimal operating condition.")

    st.markdown("### Why the data matters")
    st.markdown("""
    - Power proxy indicates how hard the machine is working; higher values can signal heavier load.
    - Temperature differential helps detect cooling efficiency and machine stress.
    - Tool wear shows accumulated degradation and maintenance urgency.
    - These summaries help identify normal behavior, degraded states, and outliers needing inspection.
    """)

    st.markdown("### Practical business value")
    st.markdown("""
    This application supports proactive maintenance by helping teams answer three questions:
    1. Is the machine operating normally?
    2. Is it entering a higher-risk wear state?
    3. Should maintenance action be scheduled now or escalated urgently?
    """)


def generate_live_sensor_reading() -> Dict[str, float]:
    """
    Generate slightly fluctuating sensor readings within realistic ranges.
    Used by the automated streamer mode.
    
    Returns:
        Dictionary of simulated sensor readings
    """
    # Start with current session state or defaults
    current = st.session_state.get('last_sensor_data', {
        'Type': np.random.choice(TYPE_OPTIONS),
        'Air temperature [K]': 300.0,
        'Process temperature [K]': 310.0,
        'Rotational speed [rpm]': 1500,
        'Torque [Nm]': 40.0,
        'Tool wear [min]': 100,
    })
    
    # Add small random fluctuations
    fluctuation_rates = {
        'Air temperature [K]': 0.3,
        'Process temperature [K]': 0.4,
        'Rotational speed [rpm]': 30,
        'Torque [Nm]': 1.5,
        'Tool wear [min]': 0.2,
    }
    
    new_reading = {}
    for key, value in current.items():
        if key == 'Type':
            new_reading[key] = np.random.choice(TYPE_OPTIONS)
        else:
            min_val, max_val = TELEMETRY_RANGES[key]
            fluctuation = np.random.normal(0, fluctuation_rates.get(key, 1))
            new_value = value + fluctuation
            new_reading[key] = np.clip(new_value, min_val, max_val)
    
    st.session_state['last_sensor_data'] = new_reading
    return new_reading


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main Streamlit application entry point.
    """
    # Page configuration
    st.set_page_config(
        page_title="Predictive Maintenance Dashboard",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Application title and description
    st.title("⚙️ Predictive Maintenance Clustering Dashboard")
    st.markdown("""
    Monitor machine telemetry to identify abnormal operating regimes, flag elevated wear risk,
    and support proactive maintenance decisions before failures escalate.
    """)

    header_col, controls_col = st.columns([3.5, 2.0])
    with header_col:
        st.caption("Use the controls to deploy the current telemetry profile and monitor the machine state.")

    with controls_col:
        enable_stream = st.checkbox(
            "Enable Live Sensor Stream",
            value=False,
            key="enable_live_stream_header",
            help="Automatically generate fluctuating sensor readings every second"
        )
        stream_speed = st.select_slider(
            "Stream Speed",
            options=[0.5, 1.0, 2.0, 5.0],
            value=1.0,
            key="stream_speed_header",
            help="Interval between sensor updates (seconds)"
        ) if enable_stream else 1.0
    
    # ========================================================================
    # MODEL LOADING
    # ========================================================================
    
    # Load models with caching and fallback
    with st.spinner("Loading ML models..."):
        pipeline = load_preprocessing_pipeline()
        cluster_model = load_cluster_model()
    
    if pipeline is None or cluster_model is None:
        st.warning(
            "⚠️ Note: Models loaded in fallback/demo mode. "
            "Real predictions will be mocked until trained models are available."
        )
    
    # ========================================================================
    # SIDEBAR CONTROLS
    # ========================================================================
    
    # Get manual telemetry input from sidebar
    sensor_data = render_sidebar_telemetry_controls()
    show_debug = st.sidebar.checkbox("Show prediction debug", value=False)
    
    # ========================================================================
    # MAIN DASHBOARD
    # ========================================================================

    tab_dashboard, tab_guidance = st.tabs(["📊 Dashboard", "🧭 How to Interpret Results"])

    with tab_dashboard:
        # Placeholder for dynamic updates
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()

        # Start the stream loop if enabled
        if enable_stream:
            st.info("📡 Live streaming enabled - sensor readings update every {:.1f}s".format(stream_speed))
            stream_container = st.empty()
            stream_log = st.empty()
            log_messages = []

            # Streaming loop
            while enable_stream:
                try:
                    # Generate new sensor reading
                    sensor_data = generate_live_sensor_reading()

                    # Make prediction
                    cluster_id, risk_label, risk_color = predict_cluster(
                        sensor_data, pipeline, cluster_model
                    )

                    # Update UI components
                    with status_placeholder.container():
                        render_operational_status(cluster_id, risk_label, risk_color)

                    with metrics_placeholder.container():
                        render_telemetry_metrics(sensor_data)

                    # Log stream event
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    log_msg = f"[{timestamp}] Cluster {cluster_id} | {risk_label}"
                    log_messages.append(log_msg)
                    log_messages = log_messages[-10:]  # Keep last 10 messages

                    with stream_log.container():
                        st.caption("📝 Stream Log (Last 10 Events):")
                        for msg in log_messages:
                            st.caption(msg)

                    # Wait before next update
                    time.sleep(stream_speed)

                except Exception as e:
                    st.error(f"Streaming error: {e}")
                    logger.exception("Error during live streaming")
                    break

        else:
            # Standard mode: display user-controlled data

            # Predict cluster for current sensor data
            if show_debug:
                cluster_id, risk_label, risk_color, debug_info = predict_cluster(
                    sensor_data, pipeline, cluster_model, debug=True
                )
            else:
                cluster_id, risk_label, risk_color = predict_cluster(
                    sensor_data, pipeline, cluster_model
                )

            # Render operational status card
            with status_placeholder.container():
                render_operational_status(cluster_id, risk_label, risk_color)

            with metrics_placeholder.container():
                render_telemetry_metrics(sensor_data)

            if show_debug:
                st.subheader("🔍 Prediction Debug")
                st.json(debug_info)

    with tab_guidance:
        render_cluster_guidance_tab()
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>🔬 Predictive Maintenance Clustering System | AI4I 2020 Dataset | DBSCAN Clustering</p>
    <p><em>Real-time machine monitoring for proactive maintenance interventions</em></p>
    <p><em>Christopher Lopez - AIM PGD AI and ML June 2025 - July 2026</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
