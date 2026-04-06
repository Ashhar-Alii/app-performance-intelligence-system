"""
streamlit_app.py - Main Streamlit Dashboard for Anomaly Detection

Author: BCA Final Year Project
Date: 2026
"""

import streamlit as st
import os
import sys
import time
import numpy as np
import pandas as pd
import hashlib

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
st.set_page_config(
    page_title="🛡️ Anomaly Detection Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        st.error("⚠️ Supabase connection failed. Check your secrets.")
        return None

supabase = init_supabase()

# ============================================================================
# AUTHENTICATION LOGIC
# ============================================================================
def hash_password(password):
    """Securely hash the password before saving/checking."""
    return hashlib.sha256(password.encode()).hexdigest()

def init_auth_state():
    if "username" not in st.session_state:
        st.session_state.username = None

init_auth_state()

# Hide app behind login wall
if st.session_state.username is None:
    st.markdown('<h1 style="text-align: center; color: #1E3A5F;">🛡️ Anomaly Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">Login to access your monitoring dashboard.</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Create Account"])
        
        with tab_login:
            log_user = st.text_input("Username", key="log_user")
            log_pass = st.text_input("Password", type="password", key="log_pass")
            if st.button("Login", use_container_width=True, type="primary"):
                if log_user and log_pass:
                    # Check DB
                    res = supabase.table("app_users").select("*").eq("username", log_user).execute()
                    if len(res.data) > 0:
                        db_hash = res.data[0]['password_hash']
                        if db_hash == hash_password(log_pass):
                            st.session_state.username = log_user
                            st.success("Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Incorrect password.")
                    else:
                        st.error("User not found.")
                else:
                    st.warning("Please enter both username and password.")

        with tab_signup:
            reg_user = st.text_input("Choose a Username", key="reg_user")
            reg_pass = st.text_input("Choose a Password", type="password", key="reg_pass")
            if st.button("Sign Up", use_container_width=True):
                if reg_user and reg_pass:
                    # Check if exists
                    check = supabase.table("app_users").select("username").eq("username", reg_user).execute()
                    if len(check.data) > 0:
                        st.error("Username already exists. Please choose another.")
                    else:
                        # Insert new user
                        supabase.table("app_users").insert({
                            "username": reg_user,
                            "password_hash": hash_password(reg_pass)
                        }).execute()
                        st.success("Account created! You can now login.")
                else:
                    st.warning("Please fill all fields.")
    
    st.stop() # Stop running the rest of the script until logged in!

# ============================================================================
# LOGGING FUNCTION (Updated for User Auth)
# ============================================================================
def log_anomaly_to_db(event, prediction, explanation=None, event_type="live"):
    """Pushes detected anomalies to DB, linked to the specific logged-in user."""
    if not supabase: return
    try:
        raw = event.get('raw_display', {})
        api_latency = raw.get('api_latency_ms', event['features'].get('api_latency_ms', 0.0))
        fps = raw.get('fps', event['features'].get('fps', 0.0))
        memory_mb = raw.get('memory_mb', event['features'].get('memory_mb', 0.0))

        data = {
            "username": st.session_state.username, # <--- NEW: Links to user
            "is_anomaly": True,
            "anomaly_score": prediction['anomaly_score_pct'],
            "severity": prediction['severity'],
            "anomaly_type": event.get('anomaly_type', event_type),
            "top_trigger": prediction['top_trigger'],
            "api_latency_ms": api_latency,
            "fps": fps,
            "memory_mb": memory_mb,
            "root_cause": explanation.get('root_cause', '') if explanation else '',
            "recommendation": explanation.get('recommendation', '') if explanation else ''
        }
        supabase.table("anomaly_logs").insert(data).execute()
    except Exception as e:
        print(f"Failed to log anomaly to DB: {e}")

# ============================================================================
# CACHED MODEL LOADING & SESSION STATE
# ============================================================================
@st.cache_resource
def load_detector():
    detector = AppAnomalyDetector(models_dir='models')
    detector.load()
    return detector

@st.cache_resource
def load_generator(_detector):
    return LiveDataGenerator(selected_features=_detector.selected_features, models_dir=os.path.join(PROJECT_ROOT, 'models'))

def init_app_state():
    if 'history' not in st.session_state: st.session_state.history = []
    if 'current_result' not in st.session_state: st.session_state.current_result = None
    if 'current_event' not in st.session_state: st.session_state.current_event = None
    if 'current_explanation' not in st.session_state: st.session_state.current_explanation = None
    if 'auto_generate' not in st.session_state: st.session_state.auto_generate = False

init_app_state()

try:
    detector = load_detector()
    generator = load_generator(detector)
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(f"### 👤 Logged in as: **{st.session_state.username}**")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.username = None
        st.rerun()
        
    st.markdown("---")
    st.markdown("## 🛡️ Control Panel")
    
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
        anomaly_type = None
        intensity = 1.0
    
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
    ai_provider = st.selectbox("Provider", ['groq', 'rule_based'], format_func=lambda x: {'rule_based': '📝 Rule-Based', 'groq': '⚡ Groq AI'}, key='ai_provider_select')
    
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
    
    if ai_provider != 'rule_based':
        if st.button("🔑 Test Connection", use_container_width=True):
            with st.spinner("Testing..."):
                result = explainer.validate_key()
                st.session_state.explainer = explainer
            if result['valid']: st.success(result['message'])
            else: st.error(result['message'])
    
    ai_status = explainer.get_status()
    if ai_status.get('rate_limited'): st.warning(ai_status['status'])
    elif ai_status.get('validated'): st.success(ai_status['status'])
    elif ai_status.get('llm_available') or ai_provider == 'rule_based': st.info(ai_status['status'])
    elif ai_status.get('error'): st.error(ai_status['status'])
    else: st.warning("⚠️ No API key configured")


# ============================================================================
# CSV UPLOAD LOGIC (Now logs anomalies to DB!)
# ============================================================================
if st.session_state.get('analyze_csv') and st.session_state.get('uploaded_csv'):
    st.session_state.analyze_csv = False
    try:
        df_upload = pd.read_csv(st.session_state.uploaded_csv)
        st.markdown("### 📁 CSV Analysis Results")
        st.caption(f"Analyzing {len(df_upload)} rows from your uploaded file...")

        csv_results = []
        for _, row in df_upload.iterrows():
            row_dict = row.to_dict()
            features = {k: float(v) for k, v in row_dict.items() if k in detector.selected_features}
            
            if len(features) < 3:
                st.warning("⚠️ CSV columns don't match model features.")
                break
            
            for f in detector.selected_features:
                if f not in features: features[f] = 0.0
                
            pred = detector.predict(features)
            
            # --- NEW: LOG CSV ANOMALIES TO DB ---
            if pred['is_anomaly']:
                # Format event block for DB function
                mock_event = {'features': features, 'raw_display': features, 'anomaly_type': 'csv_upload'}
                log_anomaly_to_db(mock_event, pred, event_type="csv_upload")
                
            csv_results.append({
                'Row': _ + 1,
                'Status': '🔴 Anomaly' if pred['is_anomaly'] else '🟢 Normal',
                'Score (%)': f"{pred['anomaly_score_pct']:.1f}%",
                'Severity': pred['severity'],
                'Top Trigger': pred['top_trigger']
            })

        if csv_results:
            results_df = pd.DataFrame(csv_results)
            total = len(results_df)
            anomalies = sum(1 for r in csv_results if 'Anomaly' in r['Status'])
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Rows", total)
            c2.metric("🔴 Anomalies Found (Saved to DB)", anomalies)
            c3.metric("Anomaly Rate", f"{anomalies/total*100:.1f}%")
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"❌ Error analyzing CSV: {e}")

# ============================================================================
# GENERATE DATA LOGIC
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
            if prediction['severity'] == 'CRITICAL': st.toast('🚨 CRITICAL anomaly detected!', icon='🔴')
    
    else:
        events = generator.generate_batch(n=50, anomaly_ratio=0.3 if force_anomaly else 0.10, anomaly_type=anomaly_type if force_anomaly else None)
        last_anomaly_index = None
        for i, event in enumerate(events):
            prediction = detector.predict(event['features'])
            explanation = explainer.explain(prediction, event, use_llm=False)
            st.session_state.history.append({'event': event, 'prediction': prediction, 'explanation': explanation})
            if prediction['is_anomaly']:
                last_anomaly_index = len(st.session_state.history) - 1
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
# MAIN DASHBOARD TABS
# ============================================================================
st.markdown('<h1 style="text-align: center; color: #1E3A5F;">🛡️ Anomaly Detection Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-top: -10px;">Real-Time Mobile App Performance Monitoring & AI-Powered Analysis</p>', unsafe_allow_html=True)
st.markdown("")

tab_live, tab_history = st.tabs(["🔴 Live Dashboard", "📈 My Historical DB Trends"])

# ──────────────────────────────────────────
# TAB 1: LIVE DASHBOARD
# ──────────────────────────────────────────
with tab_live:
    if st.session_state.current_result:
        render_metrics_row(st.session_state.current_result)
        st.markdown("")
        if st.session_state.current_event:
            render_key_metrics_cards(st.session_state.current_event)
        
        st.markdown("---")
        col_chart, col_radar = st.columns([3, 2])
        with col_chart: render_line_chart(st.session_state.history, max_points=30)
        with col_radar:
            if st.session_state.current_result.get('contributions'):
                render_radar_chart(st.session_state.current_result['contributions'], top_n=8)
        
        if len(st.session_state.history) > 10 and st.session_state.current_result.get('contributions'):
            st.markdown("---")
            st.markdown("### 📊 Feature Distribution Analysis")
            top_3_features = st.session_state.current_result['contributions'][:3]
            dist_cols = st.columns(len(top_3_features))
            for col, feat in zip(dist_cols, top_3_features):
                with col:
                    render_feature_histogram(feat['value'], feat['feature'], st.session_state.history)
        
        st.markdown("---")
        if st.session_state.current_explanation:
            explanation = st.session_state.current_explanation
            is_anomaly = st.session_state.current_result['is_anomaly']
            icon = "🚨" if is_anomaly else "✅"
            provider_display = {'gemini': '🌟 Google Gemini AI', 'groq': '⚡ Groq AI', 'rule_based': '📝 Rule-Based Engine'}.get(explanation['provider'], explanation['provider'])
            
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
        # Welcome Screen
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 3rem; background: #F8F9FA; 
                        border-radius: 15px; border: 2px dashed #DEE2E6;">
                <h2 style="color: #495057;">👋 Welcome, {st.session_state.username}!</h2>
                <p style="color: #6C757D; font-size: 1.1rem;">
                    Click <b>"🚀 Generate & Detect"</b> in the sidebar<br>
                    to start monitoring for anomalies.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────
# TAB 2: HISTORICAL DATABASE TRENDS
# ──────────────────────────────────────────
with tab_history:
    if not supabase:
        st.error("⚠️ Database connection failed.")
    else:
        st.button("🔄 Refresh My Data")
        
        with st.spinner("Fetching your historical anomalies from PostgreSQL..."):
            try:
                # NEW: Filters data ONLY for the logged in user!
                response = supabase.table("anomaly_logs").select("*").eq("username", st.session_state.username).order("timestamp", desc=True).limit(500).execute()
                db_data = response.data
            except Exception as e:
                db_data = []
                st.error(f"Error fetching data: {e}")
        
        if not db_data:
            st.info(f"Hi {st.session_state.username}, you have no anomalies saved in the database yet. Run the live generator or upload a CSV!")
        else:
            df_history = pd.DataFrame(db_data)
            
            st.markdown(f"### 📊 Overall Statistics for **{st.session_state.username}**")
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Your Total Anomalies", len(df_history))
            mode_trigger = df_history['top_trigger'].mode()[0] if not df_history.empty else "N/A"
            hc2.metric("Your Most Common Trigger", mode_trigger)
            critical_count = len(df_history[df_history['severity'] == 'CRITICAL'])
            hc3.metric("Your Critical Anomalies", critical_count)
            
            st.markdown("---")
            render_historical_time_series(df_history)
            
            st.markdown("---")
            pie_col, bar_col = st.columns(2)
            with pie_col: render_historical_severity_pie(df_history)
            with bar_col: render_historical_trigger_bar(df_history)
            
            st.markdown("---")
            st.markdown("### 🗄️ Your Raw Database Logs")
            display_df = df_history[['timestamp', 'anomaly_type', 'severity', 'top_trigger', 'anomaly_score']].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)

if st.session_state.auto_generate:
    time.sleep(3)
    st.rerun()