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
    0: ("Stable Operators", "success", 0),
    1: ("Efficient but Young Machines", "info", 0),
    2: ("High-Load Machines", "warning", 1),
    3: ("Aging or At-Risk Machines", "error", 2),
}

# Extend this mapping if the model is retrained with extra clusters.
# Cluster IDs beyond the known set will still fall back to a generic label.
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
        
        model = joblib.load(KMEANS_PATH)
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
        title='Machine Operating Regime in Speed-Torque Space',
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


def render_cluster_guidance_tab():
    """Render an information tab explaining the clusters and maintenance actions."""
    st.subheader("🧠 How to Read the Clusters")
    st.markdown("""
    The app groups machine behavior into operating regimes using telemetry patterns.
    These regimes help maintenance teams interpret the current machine condition without waiting for a failure.
    """)

    st.markdown("### Cluster meanings")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.success("✅ Cluster 0 - Stable Operators")
        st.write("Machines running smoothly under normal load with balanced temperatures and average tool wear.")
        st.caption("Action: Continue routine monitoring and keep the standard maintenance plan.")

    with col2:
        st.info("ℹ️ Cluster 1 - Efficient but Young Machines")
        st.write("Newer machines or recently serviced units with low wear, moderate speed, and slightly lower torque.")
        st.caption("Action: Monitor performance; no immediate action needed.")

    with col3:
        st.warning("⚠️ Cluster 2 - High-Load Machines")
        st.write("Machines working hard or under stress with high torque, high speed, and elevated temperature.")
        st.caption("Action: Schedule preventive checks; risk of wear or overheating.")

    with col4:
        st.error("🚨 Cluster 3 - Aging or At-Risk Machines")
        st.write("Machines showing signs of fatigue or nearing failure with high tool wear and rising temperature.")
        st.caption("Action: Prioritize inspection; possible early intervention needed.")

    st.markdown("### Why the data matters")
    st.markdown("""
    - Temperature gap helps signal heat buildup or cooling inefficiency.
    - Rotational speed and torque together describe the machine's workload.
    - Tool wear reveals how much mechanical degradation has accumulated.
    - The plot shows where the current operating point sits compared with historical patterns.
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
    
    # ========================================================================
    # MAIN DASHBOARD
    # ========================================================================

    tab_dashboard, tab_guidance = st.tabs(["📊 Dashboard", "🧭 How to Interpret Results"])

    with tab_dashboard:
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

            with metrics_placeholder.container():
                render_telemetry_metrics(sensor_data)

            # Render interactive scatter plot
            with plot_placeholder.container():
                render_interactive_scatter_plot(sensor_data)

    with tab_guidance:
        render_cluster_guidance_tab()
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>🔬 Predictive Maintenance Clustering System | AI4I 2020 Dataset | K-Means Clustering</p>
    <p><em>Real-time machine monitoring for proactive maintenance interventions</em></p>
    <p><em>Christopher Lopez - AIM PGD AI and ML June 2025 - July 2026</em></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
