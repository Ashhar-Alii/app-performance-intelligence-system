"""
streamlit_app.py - Main Streamlit Dashboard for Anomaly Detection

This is the entry point for the entire dashboard application.
It integrates all components:
- Live data generation (anomaly_detector.py)
- ML model predictions (anomaly_detector.py)
- Visual dashboard (dashboard.py)
- AI explanations (ai_explainer.py)

TO RUN:
    cd project_root/
    streamlit run app/streamlit_app.py

Author: BCA Final Year Project
Date: 2026
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
    render_data_table, render_batch_summary
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
# CACHED MODEL LOADING
# WHY @st.cache_resource: Load models ONCE, reuse across reruns
# Without this, models reload on every button click (slow!)
# ============================================================================
@st.cache_resource
def load_detector():
    """Load ML models (cached — only runs once)."""
    detector = AppAnomalyDetector(models_dir='models')
    detector.load()
    return detector

@st.cache_resource
def load_generator(_detector):
    """Load data generator (cached — only runs once)."""
    return LiveDataGenerator(
        selected_features=_detector.selected_features,
        models_dir=os.path.join(PROJECT_ROOT, 'models')
    )


# ============================================================================
# SESSION STATE INITIALIZATION
# WHY: Streamlit reruns the entire script on every interaction
# session_state persists data across reruns
# ============================================================================
def init_session_state():
    """Initialize session state variables."""
    if 'history' not in st.session_state:
        st.session_state.history = []        # List of {event, prediction, explanation}
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'current_event' not in st.session_state:
        st.session_state.current_event = None
    if 'current_explanation' not in st.session_state:
        st.session_state.current_explanation = None
    if 'auto_generate' not in st.session_state:
        st.session_state.auto_generate = False

init_session_state()


# ============================================================================
# LOAD MODELS
# ============================================================================
try:
    detector = load_detector()
    generator = load_generator(detector)
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"❌ Failed to load models: {e}")
    st.info("💡 Make sure you've run `train_model.py` first and models are in the `models/` folder.")
    st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    # ── Title ──
    st.markdown("## 🛡️ Control Panel")
    st.markdown("---")
    
    # ── Data Generation Controls ──
    st.markdown("### 🎲 Data Generation")
    
    # Mode selection
    gen_mode = st.radio(
        "Generation Mode",
        ['Single Event', 'Batch (50 events)'],
        help="Single: Generate one event at a time\nBatch: Generate 50 events at once"
    )
    
    st.markdown("---")
    
    # ── Force Anomaly Controls ──
    st.markdown("### ⚠️ Force Anomaly")
    
    force_anomaly = st.toggle(
        "Force Anomaly Mode",
        value=False,
        help="When ON, every generated event will be anomalous"
    )
    
    if force_anomaly:
        anomaly_type = st.selectbox(
            "Anomaly Type",
            options=list(ANOMALY_DESCRIPTIONS.keys()),
            format_func=lambda x: f"{ANOMALY_DESCRIPTIONS[x]['icon']} {ANOMALY_DESCRIPTIONS[x]['name']}"
        )
        
        intensity = st.slider(
            "Anomaly Intensity",
            min_value=0.5,
            max_value=2.5,
            value=1.0,
            step=0.1,
            help="0.5 = Mild (subtle)\n1.0 = Moderate\n2.0+ = Severe (obvious)"
        )
        
        # Show description
        desc = ANOMALY_DESCRIPTIONS[anomaly_type]
        st.info(f"{desc['icon']} **{desc['name']}**\n\n{desc['description']}")
    else:
        anomaly_type = None
        intensity = 1.0
    
    st.markdown("---")
    
    # ── GENERATE BUTTON ──
    generate_clicked = st.button(
        "🚀 Generate & Detect",
        use_container_width=True,
        type="primary"
    )
    
    # Auto-generate toggle
    auto_generate = st.toggle(
        "🔄 Auto-Generate (every 3s)",
        value=st.session_state.auto_generate,
        help="Automatically generate new events every 3 seconds"
    )
    st.session_state.auto_generate = auto_generate

    st.markdown("---")
    st.markdown("### 📁 Upload Your App Data")
    uploaded_csv = st.file_uploader(
        "Upload CSV to analyze",
        type=['csv'],
        help="Upload your app's telemetry CSV to detect anomalies in real data"
    )
    if uploaded_csv is not None:
        st.session_state.uploaded_csv = uploaded_csv
        st.success("✅ CSV uploaded! Click 'Analyze CSV' below.")
        if st.button("🔍 Analyze CSV", use_container_width=True, type="primary"):
            st.session_state.analyze_csv = True

    st.markdown("---")
    
    # Clear history
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.current_result = None
        st.session_state.current_event = None
        st.session_state.current_explanation = None
        st.rerun()
    
    st.markdown("---")
    
        # ── AI Explainer Config ──
    st.markdown("### 🤖 AI Explainer")
    
        # REPLACE WITH:
    ai_provider = st.selectbox(
        "Provider",
        ['groq', 'rule_based'],
        format_func=lambda x: {
            'rule_based': '📝 Rule-Based (No API needed)',
            'groq': '⚡ Groq AI (Recommended)'
        }[x],
        key='ai_provider_select'
    )
    
    # Manual API key option
    user_api_key = ""
    if ai_provider != 'rule_based':
        with st.expander("🔑 Use Your Own API Key (Optional)"):
            st.caption("Leave empty to use system key")
            user_api_key = st.text_input(
                "Your API Key",
                type="password",
                help="Optional. System key is used by default.\n"
                     "Enter your own if system limit is reached.\n\n"
                     "Groq: https://console.groq.com/keys\n"
                     "Gemini: https://aistudio.google.com/app/apikey",
                key='api_key_input'
            )
    
    # Create or update explainer
    current_config = f"{ai_provider}_{user_api_key}"
    
    if ('explainer_config' not in st.session_state or 
        st.session_state.explainer_config != current_config):
        explainer = AIExplainer(
            provider=ai_provider,
            api_key=user_api_key if user_api_key else None
        )
        st.session_state.explainer = explainer
        st.session_state.explainer_config = current_config
    else:
        explainer = st.session_state.explainer
    
    # Test Connection button
    if ai_provider != 'rule_based':
        if st.button("🔑 Test Connection", use_container_width=True):
            with st.spinner("Testing..."):
                result = explainer.validate_key()
                st.session_state.explainer = explainer
            if result['valid']:
                st.success(result['message'])
            else:
                st.error(result['message'])
    
    # Show status
    ai_status = explainer.get_status()
    
    if ai_status.get('rate_limited'):
        st.warning(ai_status['status'])
        st.caption("💡 Enter your own API key above OR switch to Rule-Based")
    elif ai_status.get('validated'):
        st.success(ai_status['status'])
    elif ai_status.get('llm_available'):
        st.info(ai_status['status'])
    elif ai_provider == 'rule_based':
        st.info(ai_status['status'])
    elif ai_status.get('error'):
        st.error(ai_status['status'])
        if 'No API key' in ai_status.get('error', ''):
            if ai_provider == 'gemini':
                st.caption("💡 Add GEMINI_API_KEY to .streamlit/secrets.toml or enter manually above")
            elif ai_provider == 'groq':
                st.caption("💡 Add GROQ_API_KEY to .streamlit/secrets.toml or enter manually above")
        else:
            st.caption(f"❌ {ai_status['error']}")
    else:
        st.warning("⚠️ No API key configured")
        if ai_provider == 'gemini':
            st.caption("Add GEMINI_API_KEY to .streamlit/secrets.toml")
        elif ai_provider == 'groq':
            st.caption("Add GROQ_API_KEY to .streamlit/secrets.toml")
    
    # ── Model Info ──
    st.markdown("### 📊 Model Info")
    model_info = detector.get_model_info()
    for key, value in model_info.items():
        st.caption(f"**{key}:** {value}")
    
    # History count
    st.caption(f"**Events in history:** {len(st.session_state.history)}")


# ============================================================================
# GENERATE DATA (when button clicked or auto-generate)
# ============================================================================
# ── CSV Upload Analysis ──
if st.session_state.get('analyze_csv') and st.session_state.get('uploaded_csv'):
    st.session_state.analyze_csv = False
    try:
        df_upload = pd.read_csv(st.session_state.uploaded_csv)
        st.markdown("### 📁 CSV Analysis Results")
        st.caption(f"Analyzing {len(df_upload)} rows from your uploaded file...")

        csv_results = []
        for _, row in df_upload.iterrows():
            row_dict = row.to_dict()
            # Only keep features the model knows
            features = {
                k: float(v) for k, v in row_dict.items()
                if k in detector.selected_features
            }
            if len(features) < 3:
                st.warning("⚠️ CSV columns don't match model features. Make sure your CSV has the right column names.")
                break
            # Fill missing features with 0
            for f in detector.selected_features:
                if f not in features:
                    features[f] = 0.0
            pred = detector.predict(features)
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
            c2.metric("🔴 Anomalies Found", anomalies)
            c3.metric("Anomaly Rate", f"{anomalies/total*100:.1f}%")
            
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            st.download_button(
                label="📥 Download Results CSV",
                data=results_df.to_csv(index=False),
                file_name="csv_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"❌ Error analyzing CSV: {e}")
        st.info("💡 Make sure your CSV has numeric columns matching the model's feature names.")
should_generate = generate_clicked or st.session_state.auto_generate

if should_generate:
    # Get explainer from session state
    explainer = st.session_state.explainer
    
    if gen_mode == 'Single Event':
        # Generate single event
        event = generator.generate_event(
            force_anomaly=force_anomaly,
            anomaly_type=anomaly_type,
            intensity=intensity
        )
        
        # Run prediction
        prediction = detector.predict(event['features'])
        
        # ← FIX: Single event → use LLM (only 1 API call)
        explanation = explainer.explain(prediction, event, use_llm=True)
        
        # Store current result
        st.session_state.current_result = prediction
        st.session_state.current_event = event
        st.session_state.current_explanation = explanation
        
        # Add to history
        st.session_state.history.append({
            'event': event,
            'prediction': prediction,
            'explanation': explanation
        })
        
        # Update explainer in session state (validation status may have changed)
        st.session_state.explainer = explainer
        
        if prediction['is_anomaly'] and prediction['severity'] == 'CRITICAL':
            st.toast('🚨 CRITICAL anomaly detected!', icon='🔴')
    
    else:
        # ← FIX: Batch generation — DON'T call LLM for every event
        # Only call LLM for the LAST anomaly event (saves API quota)
        events = generator.generate_batch(
            n=50,
            anomaly_ratio=0.3 if force_anomaly else 0.10,
            anomaly_type=anomaly_type if force_anomaly else None
        )
        
        # Process all events with rule-based explanations first
        last_anomaly_index = None
        
        for i, event in enumerate(events):
            prediction = detector.predict(event['features'])
            
            # ← FIX: use_llm=False for batch → all rule-based (instant)
            explanation = explainer.explain(prediction, event, use_llm=False)
            
            st.session_state.history.append({
                'event': event,
                'prediction': prediction,
                'explanation': explanation
            })
            
            # Track last anomaly for LLM explanation
            if prediction['is_anomaly']:
                last_anomaly_index = len(st.session_state.history) - 1
        
        # ← FIX: Now call LLM ONCE for the last anomaly event only
        if last_anomaly_index is not None and explainer.llm_available:
            last_anomaly = st.session_state.history[last_anomaly_index]
            llm_explanation = explainer.explain(
                last_anomaly['prediction'], 
                last_anomaly['event'], 
                use_llm=True
            )
            # Update that event's explanation with LLM version
            st.session_state.history[last_anomaly_index]['explanation'] = llm_explanation
            
            # Update explainer status
            st.session_state.explainer = explainer
        
        # Show latest event
        if events:
            last = st.session_state.history[-1]
            st.session_state.current_result = last['prediction']
            st.session_state.current_event = last['event']
            
            # ← FIX: Show the LLM explanation if available
            if last_anomaly_index is not None:
                st.session_state.current_explanation = st.session_state.history[last_anomaly_index]['explanation']
            else:
                st.session_state.current_explanation = last['explanation']
        
        st.toast(f'✅ Generated {len(events)} events', icon='📊')


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

# ── Title ──
st.markdown(
    '<h1 style="text-align: center; color: #1E3A5F;">🛡️ Anomaly Detection Intelligence</h1>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="text-align: center; color: #666; margin-top: -10px;">'
    'Real-Time Mobile App Performance Monitoring & AI-Powered Analysis</p>',
    unsafe_allow_html=True
)
st.markdown("")

# ── ROW 1: Big Three Metrics ──
if st.session_state.current_result:
    render_metrics_row(st.session_state.current_result)
    st.markdown("")
    
    # ── Raw Metric Values ──
    if st.session_state.current_event:
        render_key_metrics_cards(st.session_state.current_event)
    
    st.markdown("---")
    
    # ── ROW 2: Charts ──
    col_chart, col_radar = st.columns([3, 2])
    
    with col_chart:
        render_line_chart(st.session_state.history, max_points=30)
    
    with col_radar:
        if st.session_state.current_result.get('contributions'):
            render_radar_chart(st.session_state.current_result['contributions'], top_n=8)
    
    # ── Feature Distribution (if enough history) ──
    if len(st.session_state.history) > 10 and st.session_state.current_result.get('contributions'):
        st.markdown("---")
        st.markdown("### 📊 Feature Distribution Analysis")
        
        top_3_features = st.session_state.current_result['contributions'][:3]
        dist_cols = st.columns(len(top_3_features))
        
        for col, feat in zip(dist_cols, top_3_features):
            with col:
                render_feature_histogram(
                    current_value=feat['value'],
                    feature_name=feat['feature'],
                    history=st.session_state.history
                )
    
    st.markdown("---")
    
    # ── ROW 3: AI Explanation ──
    if st.session_state.current_explanation:
        explanation = st.session_state.current_explanation
        
        is_anomaly = st.session_state.current_result['is_anomaly']
        icon = "🚨" if is_anomaly else "✅"
        
        provider_display = {
            'gemini': '🌟 Google Gemini AI',
            'groq': '⚡ Groq AI',
            'rule_based': '📝 Rule-Based Engine'
        }.get(explanation['provider'], explanation['provider'])
        
        with st.expander(f"{icon} AI Root Cause Analysis", expanded=is_anomaly):
            
            # Get current provider selection from sidebar
            selected_provider = st.session_state.get('ai_provider_select', 'rule_based')
            llm_error = explanation.get('llm_error', '')
            
            if explanation['provider'] in ['gemini', 'groq']:
                # LLM worked successfully
                st.markdown(f"""
                <div style="background: #1B5E20; border-left: 4px solid #4CAF50; 
                            padding: 0.5rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
                    <b style="color: white;">✅ Powered by: {provider_display}</b>
                    <span style="color: #F0F0F0;"> — Live AI Analysis</span>
                </div>
                """, unsafe_allow_html=True)
            
            elif selected_provider != 'rule_based' and llm_error:
                # User wanted LLM but it failed — show WHY
                st.markdown(f"""
                <div style="background: #FFF3E0; border-left: 4px solid #FF9800; 
                            padding: 0.5rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
                    <b>⚠️ Powered by: {provider_display}</b> — LLM unavailable, using fallback
                    <br><small style="color: #666;">Reason: {llm_error}</small>
                </div>
                """, unsafe_allow_html=True)
            
            elif selected_provider != 'rule_based':
                # User selected LLM but no error recorded
                provider_name = 'Groq' if selected_provider == 'groq' else 'Gemini'
                st.markdown(f"""
                <div style="background: #FFF3E0; border-left: 4px solid #FF9800; 
                            padding: 0.5rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
                    <b>⚠️ Powered by: {provider_display}</b>
                    <br><small style="color: #666;">💡 Click "Test Connection" in sidebar first, then generate again</small>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                # User chose rule-based intentionally
                st.markdown(f"""
                <div style="background: #E3F2FD; border-left: 4px solid #2196F3; 
                            padding: 0.5rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;">
                    <b>Powered by: {provider_display}</b>
                    <br><small style="color: #666;">💡 Select Groq or Gemini in sidebar for AI-powered insights</small>
                </div>
                """, unsafe_allow_html=True)
            
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
                    contrib_df = pd.DataFrame([
                        {
                            'Feature': f['feature_display'],
                            'Value': f['raw_display'],
                            'Z-Score': f"{f['z_score']}σ",
                            'Direction': f"{'⬆️' if f['direction']=='HIGH' else '⬇️'} {f['direction']}"
                        }
                        for f in contributions
                    ])
                    st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                
                if is_anomaly and st.session_state.current_result.get('contributions'):
                    
                    # ── Causal Chain ──
                    st.markdown("#### 🔗 Root Cause Chain")
                    st.markdown("*Tracing the anomaly back to its source:*")
                    
                    causal_chain = explainer._build_causal_chain(
                        st.session_state.current_result['contributions']
                    )
                    
                    if causal_chain:
                        for step in causal_chain:
                            if step['step'] == len(causal_chain):
                                icon = "🔴"
                            elif step['step'] == 1:
                                icon = "⚡"
                            else:
                                icon = "→"
                            
                            st.markdown(
                                f"**{icon} Step {step['step']}** ({step['component']}): "
                                f"{step['event']}"
                            )
                    
                    # ── What-If Recovery Simulation ──
                    st.markdown("#### 🔮 Recovery Simulation")
                    st.markdown("*Projected metrics if recommended actions are taken:*")
                    
                    recovery = explainer._simulate_recovery(
                        st.session_state.current_result
                    )
                    
                    if recovery:
                        rec_col1, rec_col2, rec_col3 = st.columns(3)
                        
                        rec_col1.metric(
                            "📈 Expected Improvement",
                            f"{recovery['improvement_percent']}%",
                            "metric recovery"
                        )
                        rec_col2.metric(
                            "⏱️ Est. Recovery Time",
                            recovery['estimated_recovery_time']
                        )
                        rec_col3.metric(
                            "🎯 Confidence",
                            recovery['confidence']
                        )
                        
                        # Before/After comparison
                        import plotly.graph_objects as go
                        
                        features = list(recovery['current_state'].keys())
                        current_vals = list(recovery['current_state'].values())
                        recovered_vals = list(recovery['recovered_state'].values())
                        
                        fig = go.Figure(data=[
                            go.Bar(name='Current (Anomalous)', 
                                   x=features, y=current_vals,
                                   marker_color='#FF4B4B'),
                            go.Bar(name='After Fix (Projected)', 
                                   x=features, y=recovered_vals,
                                   marker_color='#40C057')
                        ])
                        
                        fig.update_layout(
                            title='Before vs After Recovery (Z-Score Deviation)',
                            barmode='group',
                            height=300,
                            yaxis_title='Z-Score (σ)',
                            legend=dict(orientation='h', y=-0.2)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.markdown(explanation['explanation'])
    
    # ── ROW 4: Batch Summary + Data Table ──
    if len(st.session_state.history) > 1:
        st.markdown("### 📋 Session Summary")
        render_batch_summary(st.session_state.history)
        
        st.markdown("### 📋 Recent Events")
        render_data_table(st.session_state.history, max_rows=20)
    
        # ══════════════════════════════════════════════════════════════
    # ← NEW: EXAMINER VERIFICATION SECTION
    # WHY: Allows examiner to verify model is actually detecting correctly
    # Shows side-by-side: "What we generated" vs "What model detected"
    # ══════════════════════════════════════════════════════════════
    
    if len(st.session_state.history) > 5:
        st.markdown("---")
        st.markdown("### 🔬 Model Verification ")
        st.markdown(
            "*This section proves the model is genuinely detecting anomalies, "
            "not randomly guessing.*"
        )
        
        with st.expander("📋 Click to Expand Verification Details", expanded=False):
            
            # ── 1. Ground Truth vs Prediction Comparison ──
            st.markdown("#### 1️⃣ Ground Truth vs Model Prediction")
            st.markdown(
                "Each event is **generated** as either normal or anomalous. "
                "The model **independently** predicts whether it's an anomaly. "
                "If the model works correctly, these should mostly match."
            )
            
            verification_data = []
            for i, h in enumerate(st.session_state.history):
                verification_data.append({
                    'Event #': i + 1,
                    'Generated As': ('🔴 Anomaly (' + h['event'].get('anomaly_type', 'N/A') + ')' 
                                    if h['event']['is_generated_anomaly'] 
                                    else '🟢 Normal'),
                    'Model Says': ('🔴 Anomaly' if h['prediction']['is_anomaly'] 
                                  else '🟢 Normal'),
                    'Score (%)': f"{h['prediction']['anomaly_score_pct']:.1f}%",
                    'Severity': h['prediction']['severity'],
                    'Match': ('✅ Correct' if h['event']['is_generated_anomaly'] == h['prediction']['is_anomaly']
                             else '❌ Mismatch')
                })
            
            verify_df = pd.DataFrame(verification_data)
            
            # Color code the match column
            st.dataframe(verify_df, use_container_width=True, hide_index=True)
            
            # ── 2. Accuracy Statistics ──
            st.markdown("#### 2️⃣ Detection Accuracy Statistics")
            
            total = len(st.session_state.history)
            
            generated_anomalies = sum(1 for h in st.session_state.history 
                                     if h['event']['is_generated_anomaly'])
            generated_normal = total - generated_anomalies
            
            detected_anomalies = sum(1 for h in st.session_state.history 
                                    if h['prediction']['is_anomaly'])
            
            # True Positives: Generated as anomaly AND model says anomaly
            tp = sum(1 for h in st.session_state.history 
                    if h['event']['is_generated_anomaly'] and h['prediction']['is_anomaly'])
            
            # True Negatives: Generated as normal AND model says normal
            tn = sum(1 for h in st.session_state.history 
                    if not h['event']['is_generated_anomaly'] and not h['prediction']['is_anomaly'])
            
            # False Positives: Generated as normal BUT model says anomaly
            fp = sum(1 for h in st.session_state.history 
                    if not h['event']['is_generated_anomaly'] and h['prediction']['is_anomaly'])
            
            # False Negatives: Generated as anomaly BUT model says normal
            fn = sum(1 for h in st.session_state.history 
                    if h['event']['is_generated_anomaly'] and not h['prediction']['is_anomaly'])
            
            # Display metrics
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            stat_col1.metric(
                "✅ Correctly Detected",
                f"{tp + tn}",
                f"{(tp + tn) / total * 100:.1f}% accuracy"
            )
            stat_col2.metric(
                "🎯 True Positives",
                f"{tp}",
                f"Anomalies correctly caught"
            )
            stat_col3.metric(
                "❌ False Positives", 
                f"{fp}",
                f"Normal wrongly flagged"
            )
            stat_col4.metric(
                "⚠️ Missed Anomalies",
                f"{fn}",
                f"Anomalies not caught"
            )
            
            # Detection rate
            if generated_anomalies > 0:
                detection_rate = (tp / generated_anomalies) * 100
                bg_color = '#1B5E20' if detection_rate > 70 else '#B71C1C'
                st.markdown(f"""
                <div style="background: {bg_color}; 
                            padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <b style="color: white; font-size: 1rem;">📊 Anomaly Detection Rate: {detection_rate:.1f}%</b>
                    <br>
                    <span style="color: #F0F0F0;">Out of {generated_anomalies} generated anomalies, 
                    the model correctly detected {tp}.
                    {'✅ Good performance!' if detection_rate > 70 else '⚠️ Some anomalies were subtle and missed.'}</span>
                </div>
                """, unsafe_allow_html=True)
            
            if generated_normal > 0:
                precision_rate = (tp / detected_anomalies * 100) if detected_anomalies > 0 else 0
                bg_color2 = '#1B5E20' if precision_rate > 50 else '#B71C1C'
                st.markdown(f"""
                <div style="background: {bg_color2}; 
                            padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                    <b style="color: white; font-size: 1rem;">🎯 Alert Precision: {precision_rate:.1f}%</b>
                    <br>
                    <span style="color: #F0F0F0;">Out of {detected_anomalies} alerts raised, 
                    {tp} were actual anomalies.
                    {'✅ Reliable alerts!' if precision_rate > 50 else '⚠️ Some false alarms present.'}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # ── 3. Confusion Matrix ──
            st.markdown("#### 3️⃣ Live Confusion Matrix")
            st.markdown(
                "This matrix shows the model's performance on the events "
                "generated in this session."
            )
            
            import plotly.graph_objects as go
            
            cm_fig = go.Figure(data=go.Heatmap(
                z=[[tn, fp], [fn, tp]],
                x=['Predicted Normal', 'Predicted Anomaly'],
                y=['Actually Normal', 'Actually Anomaly'],
                text=[[f'TN: {tn}', f'FP: {fp}'], [f'FN: {fn}', f'TP: {tp}']],
                texttemplate='%{text}',
                textfont=dict(size=16),
                colorscale='Blues',
                showscale=False
            ))
            
            cm_fig.update_layout(
                title='Live Confusion Matrix',
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(cm_fig, use_container_width=True)
            
            # ── 4. Score Comparison ──
            st.markdown("#### 4️⃣ Anomaly Score Comparison")
            st.markdown(
                "This chart proves the model gives **LOWER scores** to anomalous events "
                "and **HIGHER scores** to normal events. If they separate well, "
                "the model is working correctly."
            )
            
            normal_scores = [h['prediction']['anomaly_score_pct'] 
                           for h in st.session_state.history 
                           if not h['event']['is_generated_anomaly']]
            anomaly_scores = [h['prediction']['anomaly_score_pct'] 
                            for h in st.session_state.history 
                            if h['event']['is_generated_anomaly']]
            
            if normal_scores and anomaly_scores:
                import plotly.graph_objects as go
                
                score_fig = go.Figure()
                
                score_fig.add_trace(go.Box(
                    y=normal_scores,
                    name='Generated Normal',
                    marker_color='#40C057',
                    boxmean=True
                ))
                
                score_fig.add_trace(go.Box(
                    y=anomaly_scores,
                    name='Generated Anomaly',
                    marker_color='#FF4B4B',
                    boxmean=True
                ))
                
                score_fig.add_hline(
                    y=50, line_dash="dash", line_color="black",
                    annotation_text="Detection Threshold (50%)"
                )
                
                score_fig.update_layout(
                    title='Score Distribution: Normal vs Anomaly Events',
                    yaxis_title='Anomaly Score (%)',
                    height=350,
                    showlegend=True
                )
                
                st.plotly_chart(score_fig, use_container_width=True)
                
                avg_normal = np.mean(normal_scores)
                avg_anomaly = np.mean(anomaly_scores)
                
                st.markdown(f"""
                <div style="background: #1E3A5F; padding: 1rem; border-radius: 8px;">
                    <b style="color: white; font-size: 1rem;">📊 Score Analysis:</b><br>
                    <span style="color: #A8C8E8;">• Average score for Normal events: </span>
                    <b style="color: #69DB7C;">{avg_normal:.1f}%</b><br>
                    <span style="color: #A8C8E8;">• Average score for Anomaly events: </span>
                    <b style="color: #FF6B6B;">{avg_anomaly:.1f}%</b><br>
                    <span style="color: #A8C8E8;">• Score gap: </span>
                    <b style="color: white;">{abs(avg_anomaly - avg_normal):.1f}%  
                    {'✅ Good separation!' if abs(avg_anomaly - avg_normal) > 15 else '⚠️ Moderate separation'}</b>
                </div>
                """, unsafe_allow_html=True)
            
    
    # ── Download Button ──
    if st.session_state.history:
        st.markdown("---")
        
        # Prepare download data
        download_data = []
        for h in st.session_state.history:
            row = {
                'Timestamp': h['event']['timestamp'],
                'Is Anomaly': h['prediction']['is_anomaly'],
                'Score (%)': f"{h['prediction']['anomaly_score_pct']:.1f}",
                'Severity': h['prediction']['severity'],
                'Top Trigger': h['prediction']['top_trigger'],
                'Type': h['event'].get('anomaly_type', 'normal'),
            }
            for feat in ['api_latency_ms', 'fps', 'memory_mb']:
                if feat in h['event'].get('raw_display', {}):
                    row[get_display_name(feat)] = f"{h['event']['raw_display'][feat]:.0f}"
            download_data.append(row)
        
        download_df = pd.DataFrame(download_data)
        
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=download_df.to_csv(index=False),
            file_name="anomaly_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    # No data yet — show welcome screen
    st.markdown("")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #F8F9FA; 
                    border-radius: 15px; border: 2px dashed #DEE2E6;">
            <h2 style="color: #495057;">👋 Welcome!</h2>
            <p style="color: #6C757D; font-size: 1.1rem;">
                Click <b>"🚀 Generate & Detect"</b> in the sidebar<br>
                to start monitoring for anomalies.
            </p>
            <p style="color: #ADB5BD; font-size: 0.9rem; margin-top: 1rem;">
                You can also toggle <b>"Force Anomaly Mode"</b><br>
                to simulate different types of system issues.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("---")
    
    # Quick feature overview
    st.markdown("### 🔧 What This Dashboard Does")
    
    feat_cols = st.columns(4)
    features = [
        ("🎲", "Live Data", "Generate simulated app telemetry events in real-time"),
        ("🤖", "ML Detection", "Isolation Forest model trained on 10,000 events"),
        ("📊", "Visual Analysis", "Interactive charts showing anomaly patterns"),
        ("🧠", "AI Insights", "AI-powered root cause analysis and recommendations")
    ]
    
    for col, (icon, title, desc) in zip(feat_cols, features):
        with col:
            st.markdown(f"""
            <div style="text-align: center; padding: 1.5rem; background: #1E3A5F;
                        border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                        height: 160px;">
                <p style="font-size: 2rem; margin: 0;">{icon}</p>
                <p style="font-weight: 700; margin: 0.3rem 0; color: white;">{title}</p>
                <p style="font-size: 0.85rem; color: #A8C8E8;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# AUTO-GENERATE LOGIC
# ============================================================================
if st.session_state.auto_generate:
    time.sleep(3)
    st.rerun()


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #999; font-size: 0.8rem;">'
    '🛡️ Anomaly Detection Intelligence System'
    '</p>',
    unsafe_allow_html=True
)