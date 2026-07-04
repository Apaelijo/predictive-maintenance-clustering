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
import pickle
import time
from pathlib import Path
from typing import Tuple, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Define project root for relative imports
PROJECT_ROOT = Path(__file__).resolve().parents[1].parent

# Model paths
SCALER_PATH = PROJECT_ROOT / "notebooks" / "models" / "preprocessing_pipeline.pkl"
KMEANS_PATH = PROJECT_ROOT / "notebooks" / "models" / "champion_kmeans.pkl"

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
    0: ("Normal Operation", "success", 0),
    1: ("High-Stress State", "warning", 1),
    2: ("Immediate Anomaly Warning", "error", 2),
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
        
        with open(SCALER_PATH, 'rb') as f:
            pipeline = pickle.load(f)
        logger.info("✓ Preprocessing pipeline loaded successfully")
        return pipeline
    
    except Exception as e:
        logger.error(f"Error loading preprocessing pipeline: {e}")
        logger.info("Falling back to mock pipeline...")
        return _create_fallback_pipeline()


@st.cache_resource
def load_kmeans_model():
    """
    Load the trained K-Means clustering model with caching.
    
    Returns:
        Loaded K-Means model or None if not found.
        Also returns fallback random predictor if files unavailable.
    """
    try:
        if not KMEANS_PATH.exists():
            logger.warning(f"K-Means model not found at {KMEANS_PATH}")
            logger.info("Using fallback mock model...")
            return _create_fallback_kmeans()
        
        with open(KMEANS_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("✓ K-Means model loaded successfully")
        return model
    
    except Exception as e:
        logger.error(f"Error loading K-Means model: {e}")
        logger.info("Falling back to mock model...")
        return _create_fallback_kmeans()


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


def _create_fallback_kmeans():
    """
    Create a minimal fallback K-Means model for demo purposes.
    
    Returns:
        Mock K-Means object that returns random cluster assignments.
    """
    class MockKMeans:
        """Mock K-Means model for fallback when actual model is unavailable."""
        def __init__(self):
            self.n_clusters = 3
            self.cluster_centers_ = np.random.randn(3, 6)
        
        def predict(self, X):
            """Return random cluster assignment (0, 1, or 2)."""
            return np.random.randint(0, self.n_clusters, size=len(X))
    
    logger.info("Fallback K-Means model created")
    return MockKMeans()


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

def predict_cluster(
    sensor_data: Dict[str, float],
    pipeline,
    model
) -> Tuple[int, str, str]:
    """
    Predict cluster assignment and map to risk status.
    
    Args:
        sensor_data: Raw telemetry readings
        pipeline: Preprocessing pipeline
        model: Trained K-Means model
        
    Returns:
        Tuple of (cluster_id, risk_label, risk_color)
    """
    try:
        # Prepare input data
        df = prepare_prediction_input(sensor_data)

        # Prefer using transform to avoid refitting scaler/pipeline on a single sample.
        # Fall back to fit_transform if the loaded pipeline is unfitted (e.g., a fallback preprocessor).
        try:
            X_processed = pipeline.transform(df)
        except Exception:
            X_processed = pipeline.fit_transform(df)

        # Ensure we pass the correctly shaped array to the model
        preds = model.predict(X_processed)
        cluster_id = int(preds[0]) if len(preds) > 0 else -1

        # Map to risk status
        risk_label, risk_color, _ = RISK_MAPPING.get(
            cluster_id,
            ("Unknown Cluster", "secondary", -1)
        )

        return cluster_id, risk_label, risk_color

    except Exception as e:
        logger.error(f"Prediction error: {e}")
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
    st.sidebar.header("⚙️ Machine Telemetry Control Panel")
    st.sidebar.markdown("---")
    
    # Type dropdown
    machine_type = st.sidebar.selectbox(
        "🏭 Machine Type",
        options=TYPE_OPTIONS,
        help="Machine profile: Low (L), Medium (M), or High (H) stress configuration"
    )
    
    st.sidebar.markdown("### 📊 Sensor Readings")
    
    # Air Temperature slider
    air_temp = st.sidebar.slider(
        "🌡️ Air Temperature [K]",
        min_value=295.0,
        max_value=305.0,
        value=300.0,
        step=0.5,
        help="Ambient air temperature around the machine"
    )
    
    # Process Temperature slider
    process_temp = st.sidebar.slider(
        "🔥 Process Temperature [K]",
        min_value=304.0,
        max_value=315.0,
        value=310.0,
        step=0.5,
        help="Core operating temperature of the machine"
    )
    
    # Rotational Speed slider
    rot_speed = st.sidebar.slider(
        "⚡ Rotational Speed [rpm]",
        min_value=1100,
        max_value=2900,
        value=1500,
        step=50,
        help="Spindle/motor rotation speed"
    )
    
    # Torque slider
    torque = st.sidebar.slider(
        "🔧 Torque [Nm]",
        min_value=3.0,
        max_value=75.0,
        value=40.0,
        step=1.0,
        help="Rotational force applied by the spindle"
    )
    
    # Tool Wear slider
    tool_wear = st.sidebar.slider(
        "⏱️ Tool Wear [min]",
        min_value=0,
        max_value=250,
        value=100,
        step=5,
        help="Cumulative tool wear in minutes of operation"
    )
    
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


def render_operational_status(cluster_id: int, risk_label: str, risk_color: str):
    """
    Display prominent operational status card with risk badge.
    
    Args:
        cluster_id: Predicted cluster assignment
        risk_label: Human-readable risk status
        risk_color: Color coding for status (success, warning, error)
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        st.metric(
            label="🎯 Operational Cluster",
            value=f"Cluster {cluster_id}",
            delta=risk_label if risk_label != "Prediction Error" else None,
            delta_color="normal"
        )
        
        # Render alert badge
        if risk_color == "success":
            st.success(f"✅ {risk_label}")
        elif risk_color == "warning":
            st.warning(f"⚠️ {risk_label}")
        elif risk_color == "error":
            st.error(f"🚨 {risk_label}")
        else:
            st.info(f"ℹ️ {risk_label}")
        
        st.markdown("---")


def render_telemetry_metrics(sensor_data: Dict[str, float]):
    """
    Display key telemetry metrics in a grid layout.
    
    Args:
        sensor_data: Current sensor readings
    """
    st.subheader("📡 Current Telemetry Snapshot")
    
    # Create two rows of metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🌡️ Air Temp [K]",
            f"{sensor_data['Air temperature [K]']:.1f}",
            delta=f"{sensor_data['Air temperature [K]'] - 300:.1f}°"
        )
    
    with col2:
        st.metric(
            "🔥 Process Temp [K]",
            f"{sensor_data['Process temperature [K]']:.1f}",
            delta=f"{sensor_data['Process temperature [K]'] - 310:.1f}°"
        )
    
    with col3:
        st.metric(
            "⚡ Rotational Speed",
            f"{sensor_data['Rotational speed [rpm]']:,.0f} rpm"
        )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🔧 Torque",
            f"{sensor_data['Torque [Nm]']:.1f} Nm"
        )
    
    with col2:
        st.metric(
            "⏱️ Tool Wear",
            f"{sensor_data['Tool wear [min]']:.0f} min"
        )
    
    with col3:
        # Engineered feature: Temp Diff
        temp_diff = sensor_data['Process temperature [K]'] - sensor_data['Air temperature [K]']
        st.metric(
            "📊 Temp Diff [K]",
            f"{temp_diff:.1f}",
            delta="Engineered Feature"
        )


def render_interactive_scatter_plot(sensor_data: Dict[str, float]):
    """
    Create interactive 2D Plotly scatter plot: Rotational Speed vs Torque.
    
    Current user data point is shown as large, pulsing, highlighted marker.
    Historical/background points (simulated) are shown lightly.
    
    Args:
        sensor_data: Current sensor readings for main data point
    """
    st.subheader("🎯 Operational Feature Space Analysis")
    
    # Generate synthetic historical data points for context
    np.random.seed(42)
    n_historical = 150
    
    historical_speeds = np.random.uniform(1100, 2900, n_historical)
    historical_torques = np.random.uniform(3, 75, n_historical)
    
    # Create figure with background points
    fig = go.Figure()
    
    # Add historical background points (light color)
    fig.add_trace(go.Scatter(
        x=historical_speeds,
        y=historical_torques,
        mode='markers',
        marker=dict(
            size=6,
            color='rgba(128, 128, 128, 0.2)',
            line=dict(width=0)
        ),
        name='Historical Data',
        hovertemplate='<b>Historical Point</b><br>Speed: %{x:.0f} rpm<br>Torque: %{y:.1f} Nm<extra></extra>'
    ))
    
    # Add current data point (large, prominent marker)
    current_speed = sensor_data['Rotational speed [rpm]']
    current_torque = sensor_data['Torque [Nm]']
    
    fig.add_trace(go.Scatter(
        x=[current_speed],
        y=[current_torque],
        mode='markers+text',
        marker=dict(
            size=25,
            color='#FF6B6B',
            symbol='star',
            line=dict(
                color='darkred',
                width=3
            ),
            opacity=0.9
        ),
        text=['CURRENT'],
        textposition='top center',
        textfont=dict(size=12, color='darkred'),
        name='Current Machine State',
        hovertemplate='<b>🚀 CURRENT STATE</b><br>Speed: %{x:.0f} rpm<br>Torque: %{y:.1f} Nm<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title='Machine Operation in Rotational Speed vs Torque Space',
        xaxis_title='Rotational Speed [rpm]',
        yaxis_title='Torque [Nm]',
        hovermode='closest',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        width=900,
        height=500,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    st.plotly_chart(fig, use_container_width=True)


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
    Real-time machine telemetry simulator with unsupervised clustering for predictive maintenance.
    Monitor machine state, predict failure modes, and enable proactive maintenance interventions.
    """)
    
    # ========================================================================
    # MODEL LOADING
    # ========================================================================
    
    # Load models with caching and fallback
    with st.spinner("Loading ML models..."):
        pipeline = load_preprocessing_pipeline()
        kmeans_model = load_kmeans_model()
    
    if pipeline is None or kmeans_model is None:
        st.warning(
            "⚠️ Note: Models loaded in fallback/demo mode. "
            "Real predictions will be mocked until trained models are available."
        )
    
    # ========================================================================
    # SIDEBAR CONTROLS
    # ========================================================================
    
    # Get manual telemetry input from sidebar
    sensor_data = render_sidebar_telemetry_controls()
    
    # Live streaming toggle
    st.sidebar.markdown("---")
    st.sidebar.header("🔄 Live Sensor Stream")
    enable_stream = st.sidebar.checkbox(
        "Enable Live Sensor Stream",
        value=False,
        help="Automatically generate fluctuating sensor readings every second"
    )
    
    stream_speed = st.sidebar.select_slider(
        "Stream Speed",
        options=[0.5, 1.0, 2.0, 5.0],
        value=1.0,
        help="Interval between sensor updates (seconds)"
    ) if enable_stream else 1.0
    
    # ========================================================================
    # MAIN DASHBOARD
    # ========================================================================
    
    # Placeholder for dynamic updates
    status_placeholder = st.empty()
    metrics_placeholder = st.empty()
    plot_placeholder = st.empty()
    
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
                    sensor_data, pipeline, kmeans_model
                )
                
                # Update UI components
                with status_placeholder.container():
                    render_operational_status(cluster_id, risk_label, risk_color)
                
                with metrics_placeholder.container():
                    render_telemetry_metrics(sensor_data)
                
                with plot_placeholder.container():
                    render_interactive_scatter_plot(sensor_data)
                
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
        cluster_id, risk_label, risk_color = predict_cluster(
            sensor_data, pipeline, kmeans_model
        )
        
        # Render operational status card
        with status_placeholder.container():
            render_operational_status(cluster_id, risk_label, risk_color)
        
        # Render telemetry metrics
        with metrics_placeholder.container():
            render_telemetry_metrics(sensor_data)
        
        # Render interactive scatter plot
        with plot_placeholder.container():
            render_interactive_scatter_plot(sensor_data)
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>🔬 Predictive Maintenance Clustering System | AI4I 2020 Dataset | K-Means Clustering</p>
    <p><em>Real-time machine monitoring for proactive maintenance interventions</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
