"""
streamlit_app.py - Main Streamlit Dashboard for Anomaly Detection
"""

import streamlit as st
import os
import sys
import time
import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.utils import (
    inject_custom_css, ANOMALY_DESCRIPTIONS, get_severity_icon,
    get_display_name, SEVERITY_CONFIG
)
from app.anomaly_detector import AppAnomalyDetector, LiveDataGenerator
from app.dashboard import (
    render_metrics_row, render_line_chart, render_radar_chart,
    render_feature_histogram, render_key_metrics_cards,
    render_data_table, render_batch_summary,
    render_historical_time_series, render_historical_severity_pie,
    render_historical_trigger_bar
)
from app.ai_explainer import AIExplainer

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="🛡️ Anomaly Detection Intelligence", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
inject_custom_css()

# ============================================================================
# SUPABASE INITIALIZATION
# ============================================================================
from supabase import create_client, Client

@st.cache_resource
def init_supabase() -> Client | None:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
        return create_client(url, key)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Supabase not connected. DB logging disabled.")
        return None

supabase = init_supabase()

def log_anomaly_to_db(event, prediction, explanation=None):
    """Silently pushes anomalies to PostgreSQL database"""
    if not supabase: return
    try:
        raw = event.get('raw_display', {})
        api_latency = raw.get('api_latency_ms', event['features'].get('api_latency_ms', 0.0))
        fps = raw.get('fps', event['features'].get('fps', 0.0))
        memory_mb = raw.get('memory_mb', event['features'].get('memory_mb', 0.0))

        data = {
            "is_anomaly": True,
            "anomaly_score": prediction['anomaly_score_pct'],
            "severity": prediction['severity'],
            "anomaly_type": event.get('anomaly_type', 'unknown'),
            "top_trigger": prediction['top_trigger'],
            "api_latency_ms": api_latency,
            "fps": fps,
            "memory_mb": memory_mb,
            "root_cause": explanation.get('root_cause', '') if explanation else '',
            "recommendation": explanation.get('recommendation', '') if explanation else ''
        }
        supabase.table("anomaly_logs").insert(data).execute()
    except Exception as e:
        print(f"Failed to log anomaly: {e}")

# ============================================================================
# CACHED MODEL LOADING
# ============================================================================
@st.cache_resource
def load_detector():
    detector = AppAnomalyDetector(models_dir='models')
    detector.load()
    return detector

@st.cache_resource
def load_generator(_detector):
    return LiveDataGenerator(selected_features=_detector.selected_features, models_dir=os.path.join(PROJECT_ROOT, 'models'))

def init_session_state():
    if 'history' not in st.session_state: st.session_state.history = []
    if 'current_result' not in st.session_state: st.session_state.current_result = None
    if 'current_event' not in st.session_state: st.session_state.current_event = None
    if 'current_explanation' not in st.session_state: st.session_state.current_explanation = None
    if 'auto_generate' not in st.session_state: st.session_state.auto_generate = False

init_session_state()

try:
    detector = load_detector()
    generator = load_generator(detector)
    models_loaded = True
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🛡️ Control Panel")
    st.markdown("---")
    
    st.markdown("### 🎲 Data Generation")
    gen_mode = st.radio("Generation Mode", ['Single Event', 'Batch (50 events)'])
    st.markdown("---")
    
    st.markdown("### ⚠️ Force Anomaly")
    force_anomaly = st.toggle("Force Anomaly Mode", value=False)
    
    if force_anomaly:
        anomaly_type = st.selectbox("Anomaly Type", options=list(ANOMALY_DESCRIPTIONS.keys()), format_func=lambda x: f"{ANOMALY_DESCRIPTIONS[x]['icon']} {ANOMALY_DESCRIPTIONS[x]['name']}")
        intensity = st.slider("Anomaly Intensity", 0.5, 2.5, 1.0, 0.1)
        desc = ANOMALY_DESCRIPTIONS[anomaly_type]
        st.info(f"{desc['icon']} **{desc['name']}**\n\n{desc['description']}")
    else:
        anomaly_type = None; intensity = 1.0
    
    st.markdown("---")
    generate_clicked = st.button("🚀 Generate & Detect", use_container_width=True, type="primary")
    auto_generate = st.toggle("🔄 Auto-Generate (every 3s)", value=st.session_state.auto_generate)
    st.session_state.auto_generate = auto_generate
    st.markdown("---")
    
    st.markdown("### 📁 Upload Your App Data")
    uploaded_csv = st.file_uploader("Upload CSV to analyze", type=['csv'])
    if uploaded_csv is not None:
        st.session_state.uploaded_csv = uploaded_csv
        st.success("✅ CSV uploaded!")
        if st.button("🔍 Analyze CSV", use_container_width=True, type="primary"):
            st.session_state.analyze_csv = True

    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.current_result = None
        st.session_state.current_event = None
        st.session_state.current_explanation = None
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🤖 AI Explainer")
    ai_provider = st.selectbox("Provider", ['groq', 'rule_based'], format_func=lambda x: {'rule_based': '📝 Rule-Based', 'groq': '⚡ Groq AI'}[x], key='ai_provider_select')
    
    user_api_key = ""
    if ai_provider != 'rule_based':
        with st.expander("🔑 Use Your Own API Key (Optional)"):
            user_api_key = st.text_input("Your API Key", type="password", key='api_key_input')
    
    current_config = f"{ai_provider}_{user_api_key}"
    if ('explainer_config' not in st.session_state or st.session_state.explainer_config != current_config):
        explainer = AIExplainer(provider=ai_provider, api_key=user_api_key if user_api_key else None)
        st.session_state.explainer = explainer
        st.session_state.explainer_config = current_config
    else:
        explainer = st.session_state.explainer

# ============================================================================
# GENERATE DATA
# ============================================================================
should_generate = generate_clicked or st.session_state.auto_generate

if should_generate:
    explainer = st.session_state.explainer
    
    if gen_mode == 'Single Event':
        event = generator.generate_event(force_anomaly=force_anomaly, anomaly_type=anomaly_type, intensity=intensity)
        prediction = detector.predict(event['features'])
        explanation = explainer.explain(prediction, event, use_llm=True)
        
        st.session_state.current_result = prediction
        st.session_state.current_event = event
        st.session_state.current_explanation = explanation
        st.session_state.history.append({'event': event, 'prediction': prediction, 'explanation': explanation})
        st.session_state.explainer = explainer
        
        if prediction['is_anomaly']:
            log_anomaly_to_db(event, prediction, explanation)
            if prediction['severity'] == 'CRITICAL':
                st.toast('🚨 CRITICAL anomaly detected!', icon='🔴')
    
    else:
        events = generator.generate_batch(n=50, anomaly_ratio=0.3 if force_anomaly else 0.10, anomaly_type=anomaly_type if force_anomaly else None)
        last_anomaly_index = None
        
        for i, event in enumerate(events):
            prediction = detector.predict(event['features'])
            explanation = explainer.explain(prediction, event, use_llm=False)
            
            st.session_state.history.append({'event': event, 'prediction': prediction, 'explanation': explanation})
            
            if prediction['is_anomaly']:
                last_anomaly_index = len(st.session_state.history) - 1
                # Log to DB (rule-based explanation for now)
                log_anomaly_to_db(event, prediction, explanation)
        
        if last_anomaly_index is not None and explainer.llm_available:
            last_anomaly = st.session_state.history[last_anomaly_index]
            llm_explanation = explainer.explain(last_anomaly['prediction'], last_anomaly['event'], use_llm=True)
            st.session_state.history[last_anomaly_index]['explanation'] = llm_explanation
            st.session_state.explainer = explainer
        
        if events:
            last = st.session_state.history[-1]
            st.session_state.current_result = last['prediction']
            st.session_state.current_event = last['event']
            if last_anomaly_index is not None:
                st.session_state.current_explanation = st.session_state.history[last_anomaly_index]['explanation']
            else:
                st.session_state.current_explanation = last['explanation']
        
        st.toast(f'✅ Generated {len(events)} events', icon='📊')

# ============================================================================
# TABS AND MAIN DASHBOARD
# ============================================================================
st.markdown('<h1 style="text-align: center; color: #1E3A5F;">🛡️ Anomaly Detection Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-top: -10px;">Real-Time App Monitoring & AI-Powered Analysis</p>', unsafe_allow_html=True)
st.markdown("")

tab_live, tab_history = st.tabs(["🔴 Live Dashboard", "📈 Historical Database Trends"])

# ─────────────────────────────────────────
# TAB 1: LIVE DASHBOARD
# ─────────────────────────────────────────
with tab_live:
    if st.session_state.current_result:
        render_metrics_row(st.session_state.current_result)
        st.markdown("")
        
        if st.session_state.current_event:
            render_key_metrics_cards(st.session_state.current_event)
        
        st.markdown("---")
        col_chart, col_radar = st.columns([3, 2])
        with col_chart:
            render_line_chart(st.session_state.history, max_points=30)
        with col_radar:
            if st.session_state.current_result.get('contributions'):
                render_radar_chart(st.session_state.current_result['contributions'], top_n=8)
        
        st.markdown("---")
        if st.session_state.current_explanation:
            explanation = st.session_state.current_explanation
            is_anomaly = st.session_state.current_result['is_anomaly']
            icon = "🚨" if is_anomaly else "✅"
            provider_display = {'groq': '⚡ Groq AI', 'rule_based': '📝 Rule-Based Engine'}.get(explanation['provider'], explanation['provider'])
            
            with st.expander(f"{icon} AI Root Cause Analysis", expanded=is_anomaly):
                if is_anomaly:
                    st.markdown("#### 🔍 Root Cause")
                    st.markdown(f"> {explanation['root_cause']}")
                    st.markdown("#### ✅ Recommended Actions")
                    st.markdown(explanation['recommendation'])
                else:
                    st.markdown(explanation['explanation'])
        
        if len(st.session_state.history) > 1:
            st.markdown("### 📋 Recent Events")
            render_data_table(st.session_state.history, max_rows=20)
    else:
        st.info("👋 Welcome! Click **Generate & Detect** in the sidebar to start.")

# ─────────────────────────────────────────
# TAB 2: HISTORICAL DATABASE TRENDS
# ─────────────────────────────────────────
with tab_history:
    if not supabase:
        st.error("⚠️ Database connection failed. Please check your Supabase secrets.")
    else:
        # Add a refresh button to grab newest DB data manually
        st.button("🔄 Refresh DB Data")
        
        with st.spinner("Fetching historical anomalies from PostgreSQL..."):
            try:
                response = supabase.table("anomaly_logs").select("*").order("timestamp", desc=True).limit(500).execute()
                db_data = response.data
            except Exception as e:
                db_data = []
                st.error(f"Error fetching data: {e}")
        
        if not db_data:
            st.info("No anomalies saved in the database yet. Generate some anomalies in the Live Dashboard!")
        else:
            df_history = pd.DataFrame(db_data)
            
            st.markdown("### 📊 Overall Database Statistics")
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Total Anomalies Logged", len(df_history))
            hc2.metric("Most Common Trigger", df_history['top_trigger'].mode()[0] if not df_history.empty else "N/A")
            hc3.metric("Critical Anomalies", len(df_history[df_history['severity'] == 'CRITICAL']))
            
            st.markdown("---")
            render_historical_time_series(df_history)
            
            st.markdown("---")
            pie_col, bar_col = st.columns(2)
            with pie_col:
                render_historical_severity_pie(df_history)
            with bar_col:
                render_historical_trigger_bar(df_history)
            
            st.markdown("---")
            st.markdown("### 🗄️ Raw Database Logs")
            # Cleanup view for the table
            display_df = df_history[['timestamp', 'severity', 'top_trigger', 'anomaly_score', 'root_cause']].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============================================================================
# AUTO-GENERATE LOGIC
# ============================================================================
if st.session_state.auto_generate:
    time.sleep(3)
    st.rerun()