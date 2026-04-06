"""
dashboard.py - All visualization components for the Streamlit dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from app.utils import (
    SEVERITY_CONFIG, get_display_name, get_severity_color,
    get_severity_icon, scaled_to_raw, RAW_VALUE_MAPPING
)

def render_metrics_row(prediction_result):
    col1, col2, col3 = st.columns(3)
    
    is_anomaly = prediction_result['is_anomaly']
    score_pct = prediction_result['anomaly_score_pct']
    severity = prediction_result['severity']
    top_trigger = prediction_result['top_trigger']
    
    with col1:
        if is_anomaly:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #FF4B4B, #FF6B6B); 
                        color: white; padding: 1.2rem; border-radius: 12px; 
                        text-align: center; box-shadow: 0 4px 6px rgba(255,75,75,0.3);">
                <p style="font-size: 0.9rem; margin: 0; opacity: 0.9;">SYSTEM STATUS</p>
                <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0;">
                    🚨 ANOMALY
                </p>
                <p style="font-size: 0.85rem; margin: 0;">
                    Severity: {get_severity_icon(severity)} {severity}
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #40C057, #51CF66); 
                        color: white; padding: 1.2rem; border-radius: 12px; 
                        text-align: center; box-shadow: 0 4px 6px rgba(64,192,87,0.3);">
                <p style="font-size: 0.9rem; margin: 0; opacity: 0.9;">SYSTEM STATUS</p>
                <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0;">
                    ✅ NORMAL
                </p>
                <p style="font-size: 0.85rem; margin: 0;">All metrics within range</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        color = '#FF4B4B' if score_pct > 60 else '#FFA500' if score_pct > 30 else '#40C057'
        st.markdown(f"""
        <div style="background: white; padding: 1.2rem; border-radius: 12px; 
                    text-align: center; border: 2px solid {color};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="font-size: 0.9rem; margin: 0; color: #666;">ANOMALY SCORE</p>
            <p style="font-size: 2.5rem; font-weight: 800; margin: 0.3rem 0; color: {color};">
                {score_pct:.1f}%
            </p>
            <p style="font-size: 0.8rem; margin: 0; color: #999;">
                Raw: {prediction_result['anomaly_score']:.4f}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        trigger_color = '#FF4B4B' if is_anomaly else '#40C057'
        contributions = prediction_result.get('contributions', [])
        top_z = contributions[0]['z_score'] if contributions else 0
        
        st.markdown(f"""
        <div style="background: white; padding: 1.2rem; border-radius: 12px; 
                    text-align: center; border: 2px solid {trigger_color};
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <p style="font-size: 0.9rem; margin: 0; color: #666;">TOP TRIGGER FACTOR</p>
            <p style="font-size: 1.3rem; font-weight: 700; margin: 0.3rem 0; color: {trigger_color};">
                {top_trigger}
            </p>
            <p style="font-size: 0.8rem; margin: 0; color: #999;">
                Z-Score: {top_z:.1f}σ deviation
            </p>
        </div>
        """, unsafe_allow_html=True)

def render_line_chart(history, max_points=30):
    if not history:
        st.info("📊 Generate data to see the real-time chart")
        return
    recent = history[-max_points:]
    df = pd.DataFrame({
        'Event #': range(1, len(recent) + 1),
        'Anomaly Score (%)': [h['prediction']['anomaly_score_pct'] for h in recent],
        'Is Anomaly': [h['prediction']['is_anomaly'] for h in recent],
        'Timestamp': [h['event']['timestamp'] for h in recent],
        'Severity': [h['prediction']['severity'] for h in recent],
    })
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Event #'], y=df['Anomaly Score (%)'], mode='lines+markers', name='Anomaly Score',
        line=dict(color='#3498DB', width=2),
        marker=dict(size=8, color=['#FF4B4B' if a else '#3498DB' for a in df['Is Anomaly']], line=dict(width=1, color='white')),
        hovertemplate='<b>Event #%{x}</b><br>Score: %{y:.1f}%<br>Time: %{customdata}<extra></extra>', customdata=df['Timestamp']
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="red", line_width=1, annotation_text="Anomaly Threshold", annotation_position="top right", annotation_font_color="red")
    for i, row in df.iterrows():
        if row['Is Anomaly']:
            fig.add_vrect(x0=row['Event #'] - 0.4, x1=row['Event #'] + 0.4, fillcolor="rgba(255,75,75,0.1)", line_width=0)
    fig.update_layout(title=dict(text='📈 Real-Time Anomaly Score Trend', font=dict(size=16)), xaxis_title='Event Number', yaxis_title='Anomaly Score (%)', yaxis=dict(range=[0, 105]), height=350, margin=dict(l=40, r=20, t=50, b=40), plot_bgcolor='rgba(248,249,250,1)', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_radar_chart(contributions, top_n=8):
    if not contributions:
        st.info("No feature data available")
        return
    top_features = contributions[:top_n]
    categories = [f['feature_display'] for f in top_features]
    values = [min(f['z_score'], 5) for f in top_features]
    categories.append(categories[0])
    values.append(values[0])
    normal_values = [1.0] * len(categories)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=normal_values, theta=categories, fill='toself', fillcolor='rgba(64,192,87,0.15)', line=dict(color='rgba(64,192,87,0.5)', dash='dash'), name='Normal Range'))
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', fillcolor='rgba(255,75,75,0.2)', line=dict(color='#FF4B4B', width=2), name='Current Reading', marker=dict(size=6)))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=9)), angularaxis=dict(tickfont=dict(size=10))), title=dict(text='🎯 System Health Radar', font=dict(size=16)), showlegend=True, legend=dict(x=0.0, y=-0.15, orientation='h'), height=400, margin=dict(l=40, r=40, t=50, b=60))
    st.plotly_chart(fig, use_container_width=True)

def render_feature_histogram(current_value, feature_name, history):
    if not history or len(history) < 5:
        st.info("Need more data points for distribution chart")
        return
    hist_values = [h['event']['features'][feature_name] for h in history if feature_name in h['event']['features']]
    if not hist_values: return
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=hist_values, name='Historical Values', marker_color='rgba(52,152,219,0.6)', nbinsx=20))
    fig.add_vline(x=current_value, line_width=3, line_dash="solid", line_color="red", annotation_text=f"Current: {current_value:.2f}", annotation_position="top", annotation_font_color="red")
    display_name = get_display_name(feature_name)
    fig.update_layout(title=dict(text=f'📊 {display_name} Distribution', font=dict(size=14)), xaxis_title=display_name, yaxis_title='Frequency', height=300, margin=dict(l=40, r=20, t=50, b=40), showlegend=False, plot_bgcolor='rgba(248,249,250,1)')
    st.plotly_chart(fig, use_container_width=True)

def render_key_metrics_cards(event_data):
    raw_display = event_data.get('raw_display', {})
    if not raw_display: return
    cols = st.columns(len(raw_display))
    metric_icons = {'api_latency_ms': '⏱️', 'ui_response_ms': '📱', 'fps': '🖥️', 'memory_mb': '💾'}
    for col, (feat, raw_val) in zip(cols, raw_display.items()):
        with col:
            icon = metric_icons.get(feat, '📊')
            display_name = get_display_name(feat)
            mapping = RAW_VALUE_MAPPING.get(feat, {})
            unit = mapping.get('unit', '')
            higher_bad = mapping.get('higher_is_bad', True)
            mean = mapping.get('mean', 0)
            is_bad = raw_val > mean * 1.5 if higher_bad else raw_val < mean * 0.7
            color = '#FF4B4B' if is_bad else '#40C057'
            st.metric(label=f"{icon} {display_name}", value=f"{raw_val:.0f} {unit}", delta=f"{'⬆️ High' if is_bad and higher_bad else '⬇️ Low' if is_bad else '✅ Normal'}", delta_color="inverse" if (is_bad and higher_bad) or (is_bad and not higher_bad) else "normal")

def render_data_table(history, max_rows=20):
    if not history:
        st.info("No events generated yet")
        return
    recent = history[-max_rows:]
    table_data = []
    for i, h in enumerate(recent):
        pred, event = h['prediction'], h['event']
        row = {'#': len(history) - max_rows + i + 1, 'Time': event['timestamp'], 'Status': '🔴 Anomaly' if pred['is_anomaly'] else '🟢 Normal', 'Score': f"{pred['anomaly_score_pct']:.1f}%", 'Severity': f"{get_severity_icon(pred['severity'])} {pred['severity']}", 'Top Trigger': pred['top_trigger'], 'Type': event.get('anomaly_type', 'normal')}
        for feat in ['api_latency_ms', 'fps', 'memory_mb']:
            if feat in event.get('raw_display', {}):
                row[get_display_name(feat)] = f"{event['raw_display'][feat]:.0f}"
        table_data.append(row)
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, height=min(400, len(table_data) * 40 + 40), hide_index=True)

def render_batch_summary(history):
    if not history: return
    total = len(history)
    anomalies = sum(1 for h in history if h['prediction']['is_anomaly'])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", total)
    col2.metric("🟢 Normal", total - anomalies)
    col3.metric("🔴 Anomalies", anomalies)
    col4.metric("Anomaly Rate", f"{anomalies/total*100:.1f}%")

# =====================================================================
# NEW: SUPABASE HISTORICAL CHARTS
# =====================================================================

def render_historical_time_series(df):
    """Line chart showing anomalies over time from the DB."""
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Group by minute to show anomaly frequency
    df_grouped = df.set_index('timestamp').resample('1min').size().reset_index(name='Anomaly Count')
    
    fig = px.area(
        df_grouped, x='timestamp', y='Anomaly Count', 
        title='📉 Anomaly Frequency Over Time',
        color_discrete_sequence=['#FF4B4B']
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='rgba(248,249,250,1)')
    st.plotly_chart(fig, use_container_width=True)

def render_historical_severity_pie(df):
    """Donut chart showing breakdown of historical severities."""
    severity_counts = df['severity'].value_counts().reset_index()
    severity_counts.columns = ['Severity', 'Count']
    
    color_map = {'CRITICAL': '#FF0000', 'HIGH': '#FF6600', 'MEDIUM': '#FFD700', 'LOW': '#00CC00'}
    
    fig = px.pie(
        severity_counts, values='Count', names='Severity', hole=0.5,
        title='⚠️ Anomaly Severity Breakdown',
        color='Severity', color_discrete_map=color_map
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

def render_historical_trigger_bar(df):
    """Horizontal bar chart showing what caused the most anomalies."""
    trigger_counts = df['top_trigger'].value_counts().reset_index().head(5)
    trigger_counts.columns = ['Top Trigger', 'Count']
    
    fig = px.bar(
        trigger_counts, x='Count', y='Top Trigger', orientation='h',
        title='🎯 Top Contributing Factors',
        color_discrete_sequence=['#3498DB']
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='rgba(248,249,250,1)')
    st.plotly_chart(fig, use_container_width=True)