"""
utils.py - Utility constants, helpers, and styling for the Streamlit dashboard.

WHY THIS FILE?
- Centralizes all constants (colors, mappings, config)
- Avoids repeating the same values across multiple files
- Easy to change theme/styling in one place

Author: BCA Final Year Project
Date: 2026
"""
import pandas as pd
import streamlit as st

# ============================================================================
# SEVERITY CONFIGURATION
# ============================================================================

SEVERITY_CONFIG = {
    'CRITICAL': {'color': '#FF0000', 'bg': '#FFE0E0', 'icon': '🔴', 'priority': 1},
    'HIGH':     {'color': '#FF6600', 'bg': '#FFF0E0', 'icon': '🟠', 'priority': 2},
    'MEDIUM':   {'color': '#FFD700', 'bg': '#FFFDE0', 'icon': '🟡', 'priority': 3},
    'LOW':      {'color': '#00CC00', 'bg': '#E0FFE0', 'icon': '🟢', 'priority': 4},
    'NORMAL':   {'color': '#808080', 'bg': '#F0F0F0', 'icon': '⚪', 'priority': 5},
}

# ============================================================================
# FEATURE DISPLAY NAMES
# Maps internal feature names → human-readable names for the dashboard
# WHY: "api_latency_ms_rolling_mean_5" is not user-friendly
# ============================================================================

FEATURE_DISPLAY_NAMES = {
    'api_latency_ms': 'API Latency',
    'ui_response_ms': 'UI Response Time',
    'fps': 'Frames Per Second',
    'memory_mb': 'Memory Usage',
    'error_code': 'Error Code',
    'network_type': 'Network Type',
    'app_version': 'App Version',
    'screen_name': 'Screen',
    'api_latency_ms_rolling_mean_5': 'Latency (5-avg)',
    'api_latency_ms_rolling_std_5': 'Latency Variability',
    'api_latency_ms_rolling_max_10': 'Latency (10-max)',
    'api_latency_ms_zscore': 'Latency Z-Score',
    'ui_response_ms_rolling_mean_5': 'UI Response (5-avg)',
    'ui_response_ms_rolling_std_5': 'UI Response Variability',
    'ui_response_ms_rolling_max_10': 'UI Response (10-max)',
    'ui_response_ms_zscore': 'UI Response Z-Score',
    'fps_rolling_mean_5': 'FPS (5-avg)',
    'fps_rolling_std_5': 'FPS Variability',
    'fps_rolling_max_10': 'FPS (10-max)',
    'fps_zscore': 'FPS Z-Score',
    'memory_mb_rolling_mean_5': 'Memory (5-avg)',
    'memory_mb_rolling_std_5': 'Memory Variability',
    'memory_mb_rolling_max_10': 'Memory (10-max)',
    'memory_mb_zscore': 'Memory Z-Score',
    'fps_change_rate': 'FPS Change Rate',
    'latency_change_rate': 'Latency Change Rate',
    'memory_growth_rate': 'Memory Growth Rate',
    'recent_error_count': 'Recent Errors',
    'latency_cv': 'Latency CV',
}

# ============================================================================
# RAW VALUE APPROXIMATION
# Convert scaled values back to approximate real-world values for display
# WHY: Users understand "450ms latency" better than "scaled value: 2.3"
# ============================================================================

RAW_VALUE_MAPPING = {
    'api_latency_ms':    {'mean': 200, 'std': 150, 'unit': 'ms',  'higher_is_bad': True},
    'ui_response_ms':    {'mean': 100, 'std': 80,  'unit': 'ms',  'higher_is_bad': True},
    'fps':               {'mean': 55,  'std': 8,   'unit': 'fps', 'higher_is_bad': False},
    'memory_mb':         {'mean': 250, 'std': 100, 'unit': 'MB',  'higher_is_bad': True},
    'error_code':        {'mean': 0.5, 'std': 1,   'unit': '',    'higher_is_bad': True},
    'recent_error_count':{'mean': 0.5, 'std': 1.5, 'unit': '',    'higher_is_bad': True},
}

# ============================================================================
# ANOMALY TYPE DESCRIPTIONS
# ============================================================================

ANOMALY_DESCRIPTIONS = {
    'memory_leak': {
        'name': 'Memory Leak',
        'description': 'Gradual increase in memory consumption without release',
        'icon': '💾',
        'affected_features': ['memory_mb', 'memory_mb_rolling_mean_5', 'memory_growth_rate']
    },
    'latency_spike': {
        'name': 'Latency Spike', 
        'description': 'Sudden increase in API response times',
        'icon': '⏱️',
        'affected_features': ['api_latency_ms', 'api_latency_ms_rolling_mean_5', 'latency_change_rate']
    },
    'fps_drop': {
        'name': 'FPS Drop',
        'description': 'Frame rate drops causing UI lag and jank',
        'icon': '🖥️',
        'affected_features': ['fps', 'fps_rolling_mean_5', 'fps_change_rate']
    },
    'error_burst': {
        'name': 'Error Burst',
        'description': 'Sudden spike in error codes and error counts',
        'icon': '❌',
        'affected_features': ['error_code', 'recent_error_count']
    },
    'api_timeout': {
        'name': 'API Timeout',
        'description': 'API calls taking extremely long or timing out',
        'icon': '🔌',
        'affected_features': ['api_latency_ms', 'ui_response_ms', 'api_latency_ms_zscore']
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_display_name(feature):
    """Get human-readable name for a feature."""
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())

def scaled_to_raw(feature, scaled_value):
    """Convert a scaled value back to approximate raw value for display."""
    if feature in RAW_VALUE_MAPPING:
        mapping = RAW_VALUE_MAPPING[feature]
        raw = scaled_value * mapping['std'] + mapping['mean']
        return max(0, raw)  # No negative values for these metrics
    return scaled_value

def format_raw_value(feature, scaled_value):
    """Format a scaled value as a human-readable raw value with unit."""
    if feature in RAW_VALUE_MAPPING:
        mapping = RAW_VALUE_MAPPING[feature]
        raw = scaled_to_raw(feature, scaled_value)
        unit = mapping['unit']
        return f"{raw:.0f} {unit}"
    return f"{scaled_value:.4f}"

def get_severity_color(severity):
    """Get the color for a severity level."""
    return SEVERITY_CONFIG.get(severity, SEVERITY_CONFIG['NORMAL'])['color']

def get_severity_icon(severity):
    """Get the icon for a severity level."""
    return SEVERITY_CONFIG.get(severity, SEVERITY_CONFIG['NORMAL'])['icon']

def inject_custom_css():
    """Inject custom CSS for better dashboard styling."""
    st.markdown("""
    <style>
        /* Main title styling */
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A5F;
            text-align: center;
            padding: 0.5rem 0;
        }
        
        /* Status cards */
        .status-card {
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .status-anomaly {
            background: linear-gradient(135deg, #FF6B6B, #FF8E8E);
            color: white;
        }
        
        .status-normal {
            background: linear-gradient(135deg, #51CF66, #69DB7C);
            color: white;
        }
        
        /* Metric value styling */
        .big-metric {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-top: 0.3rem;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #2C3E50;
            border-bottom: 2px solid #3498DB;
            padding-bottom: 0.3rem;
            margin: 1.5rem 0 1rem 0;
        }
        
        /* AI explanation box */
        .ai-box {
            background: #F8F9FA;
            border-left: 4px solid #3498DB;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin: 0.5rem 0;
        }
        
        /* Data table improvements */
        .dataframe {
            font-size: 0.85rem;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Sidebar styling */
        .sidebar-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #2C3E50;
            margin-bottom: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

# --- NEW FUNCTION TO BE ADDED AT THE BOTTOM ---

def engineer_features_from_raw_df(raw_df, baseline_stats, selected_features):
    """
    Takes a raw DataFrame with basic columns and engineers the 29 features
    required by the ML model.

    Args:
        raw_df (pd.DataFrame): DataFrame with raw columns (e.g., 'api_latency_ms').
        baseline_stats (dict): Dictionary containing the mean and std from the training data.
        selected_features (list): The exact list of 29 feature names the model was trained on.

    Returns:
        pd.DataFrame: A new DataFrame with the 29 engineered features.
    """
    df = raw_df.copy()
    
    # Define the core raw features we expect
    raw_cols = ['api_latency_ms', 'fps', 'memory_mb', 'error_count', 'ui_response_ms']
    
    # 1. Rolling Window Features
    windows = [7, 14, 21]
    for col in raw_cols:
        if col in df.columns:
            for w in windows:
                df[f'{col}_rolling_mean_{w}'] = df[col].rolling(window=w, min_periods=1).mean()
                df[f'{col}_rolling_std_{w}'] = df[col].rolling(window=w, min_periods=1).std()
                df[f'{col}_rolling_max_{w}'] = df[col].rolling(window=w, min_periods=1).max()

    # 2. Change Rate Features
    for col in raw_cols:
        if col in df.columns:
            df[f'{col}_change_rate'] = df[col].pct_change(fill_method=None).fillna(0)
    
    # 3. Z-Score Features (using baseline stats from training)
    for col in raw_cols:
        if col in df.columns and col in baseline_stats['mean']:
            mean = baseline_stats['mean'][col]
            std = baseline_stats['std'][col]
            # Avoid division by zero if a column had zero variance in training
            if std > 0:
                df[f'{col}_z_score'] = (df[col] - mean) / std
            else:
                df[f'{col}_z_score'] = 0

    # Fill any remaining NaNs that might have been created by rolling std on small periods
    df.fillna(0, inplace=True)
    
    # Create a new DataFrame with all possible engineered features
    # and then select only the ones the model was trained on.
    
    # Create columns for all selected_features if they don't exist, and fill with 0
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0.0
            
    # Ensure the final DataFrame has the exact same columns in the exact same order
    engineered_df = df[selected_features]
    
    return engineered_df