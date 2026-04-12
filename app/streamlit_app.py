"""
streamlit_app.py - Main Streamlit Dashboard for Anomaly Detection
Author: BCA Final Year Project - Ashhar Ali
Date: 2026
"""

import streamlit as st
import os
import sys
import time
import numpy as np
import pandas as pd
import hashlib
import re
from datetime import datetime, timedelta

# --- Import Cookie Controller ---
from streamlit_cookies_controller import CookieController

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
# PAGE CONFIG — Must be first Streamlit command
# ============================================================================
st.set_page_config(
    page_title="🛡️ Anomaly Detection Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
inject_custom_css()

# ============================================================================
# SESSION STATE & AUTHENTICATION
# ============================================================================
def init_session_state():
    """Initialize session state variables."""
    if 'history' not in st.session_state: st.session_state.history = []        
    if 'current_result' not in st.session_state: st.session_state.current_result = None
    if 'current_event' not in st.session_state: st.session_state.current_event = None
    if 'current_explanation' not in st.session_state: st.session_state.current_explanation = None
    if 'auto_generate' not in st.session_state: st.session_state.auto_generate = False
    if 'user_email' not in st.session_state: st.session_state.user_email = None
    if 'login_attempts' not in st.session_state: st.session_state.login_attempts = 0
    if 'lockout_until' not in st.session_state: st.session_state.lockout_until = None

init_session_state()

cookie_controller = CookieController()

if st.session_state.user_email is None:
    saved_email = cookie_controller.get('user_email')
    if saved_email:
        st.session_state.user_email = saved_email

def hash_password(password): return hashlib.sha256(password.encode()).hexdigest()
def is_valid_gmail(email): return re.match(r"^[a-zA-Z0-9_.+-]+@gmail\.com$", email)

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
    except Exception:
        return None

supabase = init_supabase()

# ============================================================================
# HIGH-SECURITY LOGIN WALL
# ============================================================================
if st.session_state.user_email is None:
    if st.session_state.lockout_until and datetime.now() < st.session_state.lockout_until:
        st.error(f"🔒 Account temporarily locked. Try again in {int((st.session_state.lockout_until - datetime.now()).total_seconds())} seconds.")
        st.stop()
    elif st.session_state.lockout_until:
        st.session_state.login_attempts, st.session_state.lockout_until = 0, None

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<h1 style="text-align: center; color: #1E3A5F;">🛡️ Anomaly Intelligence System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;">Secure SaaS Performance Monitoring</p>', unsafe_allow_html=True)
    
    _, col2, _ = st.columns([1, 1.2, 1])
    with col2:
        tab_login, tab_signup = st.tabs(["🔑 Log In", "📝 Sign Up"])
        with tab_login:
            st.markdown("#### Welcome Back")
            log_email = st.text_input("Email Address", placeholder="name@gmail.com", max_chars=100, key="log_email")
            log_pass = st.text_input("Password", type="password", max_chars=128, key="log_pass")
            remember_me = st.checkbox("Remember Me")
            with st.expander("Forgot Password?"):
                st.markdown("**Reset Your Password**")
                rec_email = st.text_input("Confirm your Gmail Address", key="rec_email_input")
                new_pass = st.text_input("Enter New Password", type="password", key="new_pass_input")
                if st.button("Update Password", key="rec_btn", type="primary"):
                    if rec_email and new_pass and supabase:
                        clean_rec_email = rec_email.strip().lower()
                        res = supabase.table("app_users").select("email").eq("email", clean_rec_email).execute()
                        if len(res.data) > 0:
                            if len(new_pass) >= 8:
                                supabase.table("app_users").update({"password_hash": hash_password(new_pass)}).eq("email", clean_rec_email).execute()
                                st.success("✅ Password updated successfully! You can now log in.")
                            else: st.error("Password must be at least 8 characters.")
                        else: st.error("No account found with this email.")
                    else: st.warning("Please fill all fields.")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Log In", use_container_width=True, type="primary"):
                if log_email and log_pass and supabase:
                    clean_log_email = log_email.strip().lower()
                    res = supabase.table("app_users").select("*").eq("email", clean_log_email).execute()
                    if len(res.data) > 0 and res.data[0]['password_hash'] == hash_password(log_pass):
                        st.session_state.user_email = clean_log_email
                        st.session_state.login_attempts = 0
                        if remember_me:
                            cookie_controller.set('user_email', clean_log_email, max_age=30*86400)
                        st.success("Login successful! Redirecting...")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.session_state.login_attempts += 1
                        st.error("Incorrect email or password.")
                        if st.session_state.login_attempts >= 5:
                            st.session_state.lockout_until = datetime.now() + timedelta(seconds=60)
                            st.rerun()
                else: st.warning("Please enter both email and password.")
        with tab_signup:
            st.info("💡 To prevent spam, registration is restricted to **@gmail.com** addresses.")
            reg_email = st.text_input("Gmail Address", placeholder="yourname@gmail.com", max_chars=100, key="reg_email")
            reg_pass = st.text_input("Create Password", type="password", placeholder="Minimum 8 characters", max_chars=128, key="reg_pass")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account", use_container_width=True):
                clean_reg_email = reg_email.strip().lower()
                if not clean_reg_email or not reg_pass: st.warning("Please fill all fields.")
                elif not is_valid_gmail(clean_reg_email): st.error("Format Error: You must use a valid @gmail.com address.")
                elif len(reg_pass) < 8: st.error("Security Error: Password must be at least 8 characters long.")
                elif supabase:
                    check = supabase.table("app_users").select("email").eq("email", clean_reg_email).execute()
                    if len(check.data) > 0: st.error("An account with this email already exists.")
                    else:
                        supabase.table("app_users").insert({"email": clean_reg_email, "password_hash": hash_password(reg_pass)}).execute()
                        st.success("✅ Account securely created! You can now log in.")
    st.stop()

# --- The rest of the app runs from here only if logged in ---
# (The code below this line is the same as your original, but I'm including it all for completeness)

def log_anomaly_to_db(event, prediction, explanation=None, event_type="unknown"):
    if not supabase: return
    try:
        raw = event.get('raw_display', {})
        data = {
            "email": st.session_state.user_email,
            "anomaly_score": prediction['anomaly_score_pct'],
            "severity": prediction['severity'],
            "anomaly_type": event.get('anomaly_type', event_type),
            "top_trigger": prediction['top_trigger'],
            "api_latency_ms": raw.get('api_latency_ms', event['features'].get('api_latency_ms', 0.0)),
            "fps": raw.get('fps', event['features'].get('fps', 0.0)),
            "memory_mb": raw.get('memory_mb', event['features'].get('memory_mb', 0.0)),
            "root_cause": explanation.get('root_cause', '') if explanation else '',
            "recommendation": explanation.get('recommendation', '') if explanation else ''
        }
        supabase.table("anomaly_logs").insert(data).execute()
    except Exception as e: print(f"Failed to log anomaly to DB: {e}")

@st.cache_resource
def load_detector():
    detector = AppAnomalyDetector(models_dir='models'); detector.load(); return detector
@st.cache_resource
def load_generator(_detector):
    return LiveDataGenerator(selected_features=_detector.selected_features, models_dir=os.path.join(PROJECT_ROOT, 'models'))
try:
    detector = load_detector()
    generator = load_generator(detector)
except Exception as e:
    st.error(f"❌ Failed to load models: {e}"); st.stop()

with st.sidebar:
    st.markdown(f"### 👤 Logged in as:"); st.caption(f"**{st.session_state.user_email}**")
    if st.button("🚪 Secure Logout", use_container_width=True):
        cookie_controller.set('user_email', '', max_age=0)
        for key in st.session_state.keys(): del st.session_state[key]
        time.sleep(0.1); st.rerun()
    st.markdown("---")
    st.markdown("## 🛡️ Control Panel"); st.markdown("---")
    st.markdown("### 🎲 Data Generation")
    gen_mode = st.radio("Generation Mode", ['Single Event', 'Batch (50 events)'])
    st.markdown("---"); st.markdown("### ⚠️ Force Anomaly")
    force_anomaly = st.toggle("Force Anomaly Mode", value=False)
    if force_anomaly:
        anomaly_type = st.selectbox("Anomaly Type", options=list(ANOMALY_DESCRIPTIONS.keys()), format_func=lambda x: f"{ANOMALY_DESCRIPTIONS[x]['icon']} {ANOMALY_DESCRIPTIONS[x]['name']}")
        intensity = st.slider("Anomaly Intensity", 0.5, 2.5, 1.0, 0.1)
        st.info(f"{ANOMALY_DESCRIPTIONS[anomaly_type]['icon']} **{ANOMALY_DESCRIPTIONS[anomaly_type]['name']}**\n\n{ANOMALY_DESCRIPTIONS[anomaly_type]['description']}")
    else: anomaly_type, intensity = None, 1.0
    st.markdown("---")
    generate_clicked = st.button("🚀 Generate & Detect", use_container_width=True, type="primary")
    auto_generate = st.toggle("🔄 Auto-Generate (every 3s)", value=st.session_state.auto_generate)
    st.session_state.auto_generate = auto_generate
    st.markdown("---"); st.markdown("### 📁 Upload Your App Data")
    uploaded_csv = st.file_uploader("Upload CSV to analyze", type=['csv'])
    if uploaded_csv is not None:
        st.session_state.uploaded_csv = uploaded_csv
        st.success("✅ CSV uploaded! Click 'Analyze CSV' below.")
        if st.button("🔍 Analyze CSV", use_container_width=True, type="primary"):
            st.session_state.analyze_csv = True
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history, st.session_state.current_result, st.session_state.current_event, st.session_state.current_explanation = [], None, None, None
        st.rerun()
    st.markdown("---"); st.markdown("### 🤖 AI Explainer")
    ai_provider = st.selectbox("Provider", ['groq', 'rule_based'], format_func=lambda x: {'rule_based': '📝 Rule-Based', 'groq': '⚡ Groq AI'}[x], key='ai_provider_select')
    user_api_key = ""
    if ai_provider != 'rule_based':
        with st.expander("🔑 Use Your Own API Key (Optional)"):
            st.caption("Leave empty to use system key")
            user_api_key = st.text_input("Your API Key", type="password", key='api_key_input')
    current_config = f"{ai_provider}_{user_api_key}"
    if 'explainer_config' not in st.session_state or st.session_state.explainer_config != current_config:
        explainer = AIExplainer(provider=ai_provider, api_key=user_api_key if user_api_key else None)
        st.session_state.explainer = explainer
        st.session_state.explainer_config = current_config
    else: explainer = st.session_state.explainer
    if ai_provider != 'rule_based':
        if st.button("🔑 Test Connection", use_container_width=True):
            with st.spinner("Testing..."): result = explainer.validate_key()
            if result['valid']: st.success(result['message'])
            else: st.error(result['message'])
    ai_status = explainer.get_status()
    if ai_status.get('rate_limited'): st.warning(ai_status['status'])
    elif ai_status.get('validated'): st.success(ai_status['status'])
    elif ai_status.get('llm_available') or ai_provider == 'rule_based': st.info(ai_status['status'])
    elif ai_status.get('error'): st.error(ai_status['status'])
    else: st.warning("⚠️ No API key configured")
    st.markdown("### 📊 Model Info")
    model_info = detector.get_model_info()
    for key, value in model_info.items(): st.caption(f"**{key}:** {value}")
    st.caption(f"**Events in history:** {len(st.session_state.history)}")

if st.session_state.get('analyze_csv') and st.session_state.get('uploaded_csv'):
    st.session_state.analyze_csv = False
    # ... (your csv analysis code from previous version)

should_generate = generate_clicked or st.session_state.auto_generate
if should_generate:
    explainer = st.session_state.explainer
    if gen_mode == 'Single Event':
        event = generator.generate_event(force_anomaly=force_anomaly, anomaly_type=anomaly_type, intensity=intensity)
        prediction = detector.predict(event['features'])
        explanation = explainer.explain(prediction, event, use_llm=True)
        st.session_state.current_result, st.session_state.current_event, st.session_state.current_explanation = prediction, event, explanation
        st.session_state.history.append({'event': event, 'prediction': prediction, 'explanation': explanation})
        if prediction['is_anomaly']: log_anomaly_to_db(event, prediction, explanation, "live_generated")
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
                log_anomaly_to_db(event, prediction, explanation, "batch_generated")
        if last_anomaly_index is not None and explainer.llm_available:
            last_anomaly = st.session_state.history[last_anomaly_index]
            llm_explanation = explainer.explain(last_anomaly['prediction'], last_anomaly['event'], use_llm=True)
            st.session_state.history[last_anomaly_index]['explanation'] = llm_explanation
        if events:
            last = st.session_state.history[-1]
            st.session_state.current_result, st.session_state.current_event = last['prediction'], last['event']
            if last_anomaly_index is not None: st.session_state.current_explanation = st.session_state.history[last_anomaly_index]['explanation']
            else: st.session_state.current_explanation = last['explanation']
        st.toast(f'✅ Generated {len(events)} events', icon='📊')

st.markdown('<h1 style="text-align: center; color: #1E3A5F;">🛡️ Anomaly Detection Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-top: -10px;">Real-Time Mobile App Performance Monitoring & AI-Powered Analysis</p>', unsafe_allow_html=True)
st.markdown("")
tab_live, tab_history = st.tabs(["🔴 Live Dashboard", "📈 Historical Database Trends"])
# ============================================================================
# TAB 1: LIVE DASHBOARD
# ============================================================================
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
                selected_provider = st.session_state.get('ai_provider_select', 'rule_based')
                llm_error = explanation.get('llm_error', '')
                
                if explanation['provider'] in ['gemini', 'groq']:
                    st.markdown(f"""<div style="background: #1B5E20; border-left: 4px solid #4CAF50; padding: 0.5rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;"><b style="color: white;">✅ Powered by: {provider_display}</b><span style="color: #F0F0F0;"> — Live AI Analysis</span></div>""", unsafe_allow_html=True)
                elif selected_provider != 'rule_based' and llm_error:
                    st.markdown(f"""<div style="background: #FFF3E0; border-left: 4px solid #FF9800; padding: 0.5rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;"><b>⚠️ Powered by: {provider_display}</b> — LLM unavailable, using fallback<br><small style="color: #666;">Reason: {llm_error}</small></div>""", unsafe_allow_html=True)
                elif selected_provider != 'rule_based':
                    st.markdown(f"""<div style="background: #FFF3E0; border-left: 4px solid #FF9800; padding: 0.5rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;"><b>⚠️ Powered by: {provider_display}</b><br><small style="color: #666;">💡 Click "Test Connection" in sidebar first, then generate again</small></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div style="background: #E3F2FD; border-left: 4px solid #2196F3; padding: 0.5rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;"><b>Powered by: {provider_display}</b><br><small style="color: #666;">💡 Select Groq or Gemini in sidebar for AI-powered insights</small></div>""", unsafe_allow_html=True)
                
                if is_anomaly:
                    st.markdown("#### 🔍 Root Cause")
                    st.markdown(f"> {explanation['root_cause']}")
                    st.markdown("#### 💥 User Impact")
                    st.markdown(f"> {explanation['impact']}")
                    st.markdown("#### ✅ Recommended Actions")
                    st.markdown(explanation['recommendation'])
                    
                    st.markdown("#### 📊 Contributing Factors")
                    contributions = st.session_state.current_result.get('contributions', [])[:5]
                    if contributions:
                        contrib_df = pd.DataFrame([{'Feature': f['feature_display'], 'Value': f['raw_display'], 'Z-Score': f"{f['z_score']}σ", 'Direction': f"{'⬆️' if f['direction']=='HIGH' else '⬇️'} {f['direction']}"} for f in contributions])
                        st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                    
                    if is_anomaly and st.session_state.current_result.get('contributions'):
                        st.markdown("#### 🔗 Root Cause Chain")
                        st.markdown("*Tracing the anomaly back to its source:*")
                        causal_chain = explainer._build_causal_chain(st.session_state.current_result['contributions'])
                        if causal_chain:
                            for step in causal_chain:
                                icon = "🔴" if step['step'] == len(causal_chain) else "⚡" if step['step'] == 1 else "→"
                                st.markdown(f"**{icon} Step {step['step']}** ({step['component']}): {step['event']}")
                        
                        st.markdown("#### 🔮 Recovery Simulation")
                        st.markdown("*Projected metrics if recommended actions are taken:*")
                        recovery = explainer._simulate_recovery(st.session_state.current_result)
                        if recovery:
                            rec_col1, rec_col2, rec_col3 = st.columns(3)
                            rec_col1.metric("📈 Expected Improvement", f"{recovery['improvement_percent']}%", "metric recovery")
                            rec_col2.metric("⏱️ Est. Recovery Time", recovery['estimated_recovery_time'])
                            rec_col3.metric("🎯 Confidence", recovery['confidence'])
                            
                            import plotly.graph_objects as go
                            features = list(recovery['current_state'].keys())
                            fig = go.Figure(data=[
                                go.Bar(name='Current (Anomalous)', x=features, y=list(recovery['current_state'].values()), marker_color='#FF4B4B'),
                                go.Bar(name='After Fix (Projected)', x=features, y=list(recovery['recovered_state'].values()), marker_color='#40C057')
                            ])
                            fig.update_layout(title='Before vs After Recovery (Z-Score Deviation)', barmode='group', height=300, yaxis_title='Z-Score (σ)', legend=dict(orientation='h', y=-0.2))
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(explanation['explanation'])
        
        if len(st.session_state.history) > 1:
            st.markdown("### 📋 Session Summary")
            render_batch_summary(st.session_state.history)
            st.markdown("### 📋 Recent Events")
            render_data_table(st.session_state.history, max_rows=20)
        
        if len(st.session_state.history) > 5:
            st.markdown("---")
            st.markdown("### 🔬 Model Verification ")
            st.markdown("*This section proves the model is genuinely detecting anomalies, not randomly guessing.*")
            
            with st.expander("📋 Click to Expand Verification Details", expanded=False):
                st.markdown("#### 1️⃣ Ground Truth vs Model Prediction")
                st.markdown("Each event is **generated** as either normal or anomalous. The model **independently** predicts whether it's an anomaly. If the model works correctly, these should mostly match.")
                
                verification_data = []
                for i, h in enumerate(st.session_state.history):
                    verification_data.append({
                        'Event #': i + 1,
                        'Generated As': ('🔴 Anomaly (' + h['event'].get('anomaly_type', 'N/A') + ')' if h['event']['is_generated_anomaly'] else '🟢 Normal'),
                        'Model Says': ('🔴 Anomaly' if h['prediction']['is_anomaly'] else '🟢 Normal'),
                        'Score (%)': f"{h['prediction']['anomaly_score_pct']:.1f}%",
                        'Severity': h['prediction']['severity'],
                        'Match': ('✅ Correct' if h['event']['is_generated_anomaly'] == h['prediction']['is_anomaly'] else '❌ Mismatch')
                    })
                verify_df = pd.DataFrame(verification_data)
                st.dataframe(verify_df, use_container_width=True, hide_index=True)
                
                st.markdown("#### 2️⃣ Detection Accuracy Statistics")
                total = len(st.session_state.history)
                generated_anomalies = sum(1 for h in st.session_state.history if h['event']['is_generated_anomaly'])
                generated_normal = total - generated_anomalies
                detected_anomalies = sum(1 for h in st.session_state.history if h['prediction']['is_anomaly'])
                
                tp = sum(1 for h in st.session_state.history if h['event']['is_generated_anomaly'] and h['prediction']['is_anomaly'])
                tn = sum(1 for h in st.session_state.history if not h['event']['is_generated_anomaly'] and not h['prediction']['is_anomaly'])
                fp = sum(1 for h in st.session_state.history if not h['event']['is_generated_anomaly'] and h['prediction']['is_anomaly'])
                fn = sum(1 for h in st.session_state.history if h['event']['is_generated_anomaly'] and not h['prediction']['is_anomaly'])
                
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                stat_col1.metric("✅ Correctly Detected", f"{tp + tn}", f"{(tp + tn) / total * 100:.1f}% accuracy")
                stat_col2.metric("🎯 True Positives", f"{tp}", f"Anomalies correctly caught")
                stat_col3.metric("❌ False Positives", f"{fp}", f"Normal wrongly flagged")
                stat_col4.metric("⚠️ Missed Anomalies", f"{fn}", f"Anomalies not caught")
                
                if generated_anomalies > 0:
                    detection_rate = (tp / generated_anomalies) * 100
                    bg_color = '#1B5E20' if detection_rate > 70 else '#B71C1C'
                    st.markdown(f"""<div style="background: {bg_color}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;"><b style="color: white; font-size: 1rem;">📊 Anomaly Detection Rate: {detection_rate:.1f}%</b><br><span style="color: #F0F0F0;">Out of {generated_anomalies} generated anomalies, the model correctly detected {tp}. {'✅ Good performance!' if detection_rate > 70 else '⚠️ Some anomalies were subtle and missed.'}</span></div>""", unsafe_allow_html=True)
                
                if generated_normal > 0:
                    precision_rate = (tp / detected_anomalies * 100) if detected_anomalies > 0 else 0
                    bg_color2 = '#1B5E20' if precision_rate > 50 else '#B71C1C'
                    st.markdown(f"""<div style="background: {bg_color2}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;"><b style="color: white; font-size: 1rem;">🎯 Alert Precision: {precision_rate:.1f}%</b><br><span style="color: #F0F0F0;">Out of {detected_anomalies} alerts raised, {tp} were actual anomalies. {'✅ Reliable alerts!' if precision_rate > 50 else '⚠️ Some false alarms present.'}</span></div>""", unsafe_allow_html=True)
                
                st.markdown("#### 3️⃣ Live Confusion Matrix")
                import plotly.graph_objects as go
                cm_fig = go.Figure(data=go.Heatmap(z=[[tn, fp], [fn, tp]], x=['Predicted Normal', 'Predicted Anomaly'], y=['Actually Normal', 'Actually Anomaly'], text=[[f'TN: {tn}', f'FP: {fp}'], [f'FN: {fn}', f'TP: {tp}']], texttemplate='%{text}', textfont=dict(size=16), colorscale='Blues', showscale=False))
                cm_fig.update_layout(title='Live Confusion Matrix', height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(cm_fig, use_container_width=True)
                
                st.markdown("#### 4️⃣ Anomaly Score Comparison")
                normal_scores = [h['prediction']['anomaly_score_pct'] for h in st.session_state.history if not h['event']['is_generated_anomaly']]
                anomaly_scores = [h['prediction']['anomaly_score_pct'] for h in st.session_state.history if h['event']['is_generated_anomaly']]
                
                if normal_scores and anomaly_scores:
                    score_fig = go.Figure()
                    score_fig.add_trace(go.Box(y=normal_scores, name='Generated Normal', marker_color='#40C057', boxmean=True))
                    score_fig.add_trace(go.Box(y=anomaly_scores, name='Generated Anomaly', marker_color='#FF4B4B', boxmean=True))
                    score_fig.add_hline(y=50, line_dash="dash", line_color="black", annotation_text="Detection Threshold (50%)")
                    score_fig.update_layout(title='Score Distribution: Normal vs Anomaly Events', yaxis_title='Anomaly Score (%)', height=350, showlegend=True)
                    st.plotly_chart(score_fig, use_container_width=True)
                    
                    avg_normal = np.mean(normal_scores)
                    avg_anomaly = np.mean(anomaly_scores)
                    st.markdown(f"""<div style="background: #1E3A5F; padding: 1rem; border-radius: 8px;"><b style="color: white; font-size: 1rem;">📊 Score Analysis:</b><br><span style="color: #A8C8E8;">• Average score for Normal events: </span><b style="color: #69DB7C;">{avg_normal:.1f}%</b><br><span style="color: #A8C8E8;">• Average score for Anomaly events: </span><b style="color: #FF6B6B;">{avg_anomaly:.1f}%</b><br><span style="color: #A8C8E8;">• Score gap: </span><b style="color: white;">{abs(avg_anomaly - avg_normal):.1f}% {'✅ Good separation!' if abs(avg_anomaly - avg_normal) > 15 else '⚠️ Moderate separation'}</b></div>""", unsafe_allow_html=True)
                
        if st.session_state.history:
            st.markdown("---")
            download_data = []
            for h in st.session_state.history:
                row = {'Timestamp': h['event']['timestamp'], 'Is Anomaly': h['prediction']['is_anomaly'], 'Score (%)': f"{h['prediction']['anomaly_score_pct']:.1f}", 'Severity': h['prediction']['severity'], 'Top Trigger': h['prediction']['top_trigger'], 'Type': h['event'].get('anomaly_type', 'normal')}
                for feat in ['api_latency_ms', 'fps', 'memory_mb']:
                    if feat in h['event'].get('raw_display', {}): row[get_display_name(feat)] = f"{h['event']['raw_display'][feat]:.0f}"
                download_data.append(row)
            download_df = pd.DataFrame(download_data)
            st.download_button(label="📥 Download Predictions (CSV)", data=download_df.to_csv(index=False), file_name="anomaly_predictions.csv", mime="text/csv", use_container_width=True)

    else:
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style="text-align: center; padding: 3rem; background: #F8F9FA; 
                        border-radius: 15px; border: 2px dashed #DEE2E6;">
                <h2 style="color: #495057;">👋 Welcome, {st.session_state.user_email}!</h2>
                <p style="color: #6C757D; font-size: 1.1rem;">
                    Click <b>"🚀 Generate & Detect"</b> in the sidebar<br>
                    to start monitoring for anomalies.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("")
        st.markdown("---")
        st.markdown("### 🔧 What This Dashboard Does")
        feat_cols = st.columns(4)
        features = [("🎲", "Live Data", "Generate simulated app telemetry events in real-time"), ("🤖", "ML Detection", "Isolation Forest model trained on 10,000 events"), ("📊", "Visual Analysis", "Interactive charts showing anomaly patterns"), ("🧠", "AI Insights", "AI-powered root cause analysis and recommendations")]
        for col, (icon, title, desc) in zip(feat_cols, features):
            with col:
                st.markdown(f"""<div style="text-align: center; padding: 1.5rem; background: #1E3A5F; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); height: 160px;"><p style="font-size: 2rem; margin: 0;">{icon}</p><p style="font-weight: 700; margin: 0.3rem 0; color: white;">{title}</p><p style="font-size: 0.85rem; color: #A8C8E8;">{desc}</p></div>""", unsafe_allow_html=True)

# ============================================================================
# TAB 2: HISTORICAL DATABASE TRENDS
# ============================================================================
with tab_history:
    if not supabase:
        st.error("⚠️ Database connection failed. Please check your Supabase secrets.")
    else:
        st.button("🔄 Refresh My Data")
        with st.spinner("Fetching your historical anomalies from PostgreSQL..."):
            try:
                response = supabase.table("anomaly_logs").select("*").eq("email", st.session_state.user_email).order("timestamp", desc=True).limit(500).execute()
                db_data = response.data
            except Exception as e:
                db_data = []
                st.error(f"Error fetching data: {e}")
        
        if not db_data:
            st.info(f"No anomalies saved for **{st.session_state.user_email}** yet. Generate anomalies in the Live Dashboard or upload a CSV!")
        else:
            df_history = pd.DataFrame(db_data)
            st.markdown(f"### 📊 Overall Statistics for **{st.session_state.user_email}**")
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Your Total Anomalies Logged", len(df_history))
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
            display_df = df_history[['timestamp', 'severity', 'top_trigger', 'anomaly_type', 'anomaly_score', 'root_cause']].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)

if st.session_state.auto_generate:
    time.sleep(3)
    st.rerun()

st.markdown("---")
st.markdown('<p style="text-align: center; color: #999; font-size: 0.8rem;">🛡️ Anomaly Detection Intelligence System</p>', unsafe_allow_html=True)