"""
dashboard.py - All visualization components for the Streamlit dashboard.

Contains functions to render:
1. Big Three Metrics (Status, Score, Top Trigger)
2. Real-time Line Chart (last 30 events)
3. Radar/Spider Chart (feature health)
4. Feature Distribution Histogram
5. Data Table (recent events)
6. Batch Summary Statistics

WHY SEPARATE FILE?
- Keeps visualization logic separate from app logic
- Easy to modify chart styling without touching business logic
- Reusable components

Author: BCA Final Year Project
Date: 2026
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
    """
    Render the "Big Three" metrics at the top of the dashboard.
    
    METRICS:
    1. Anomaly Status — RED "Anomaly Detected" or GREEN "System Normal"
    2. Anomaly Score — 0-100% gauge (higher = more anomalous)
    3. Top Trigger Factor — Which feature contributed most
    
    WHY THESE THREE?
    - Status: Immediate yes/no answer
    - Score: How confident the model is
    - Trigger: What to investigate first
    """
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
        # Anomaly Score Gauge
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
        # Top Trigger Factor
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
    """
    Render real-time line chart of anomaly scores over time.
    
    Shows last N data points with:
    - Blue line for normal events
    - Red dots for anomaly events
    - Threshold line
    
    WHY LINE CHART?
    - Shows trends over time (is system degrading?)
    - Red dots immediately highlight anomalies
    - Threshold line shows the decision boundary
    """
    if not history:
        st.info("📊 Generate data to see the real-time chart")
        return
    
    # Get last N points
    recent = history[-max_points:]
    
    df = pd.DataFrame({
        'Event #': range(1, len(recent) + 1),
        'Anomaly Score (%)': [h['prediction']['anomaly_score_pct'] for h in recent],
        'Is Anomaly': [h['prediction']['is_anomaly'] for h in recent],
        'Timestamp': [h['event']['timestamp'] for h in recent],
        'Severity': [h['prediction']['severity'] for h in recent],
    })
    
    fig = go.Figure()
    
    # Main score line
    fig.add_trace(go.Scatter(
        x=df['Event #'],
        y=df['Anomaly Score (%)'],
        mode='lines+markers',
        name='Anomaly Score',
        line=dict(color='#3498DB', width=2),
        marker=dict(
            size=8,
            color=['#FF4B4B' if a else '#3498DB' for a in df['Is Anomaly']],
            line=dict(width=1, color='white')
        ),
        hovertemplate='<b>Event #%{x}</b><br>Score: %{y:.1f}%<br>Time: %{customdata}<extra></extra>',
        customdata=df['Timestamp']
    ))
    
    # Threshold line at 50% (our normalized threshold)
    fig.add_hline(
        y=50, line_dash="dash", line_color="red", line_width=1,
        annotation_text="Anomaly Threshold",
        annotation_position="top right",
        annotation_font_color="red"
    )
    
    # Highlight anomaly regions
    for i, row in df.iterrows():
        if row['Is Anomaly']:
            fig.add_vrect(
                x0=row['Event #'] - 0.4, x1=row['Event #'] + 0.4,
                fillcolor="rgba(255,75,75,0.1)", line_width=0
            )
    
    fig.update_layout(
        title=dict(text='📈 Real-Time Anomaly Score Trend', font=dict(size=16)),
        xaxis_title='Event Number',
        yaxis_title='Anomaly Score (%)',
        yaxis=dict(range=[0, 105]),
        height=350,
        margin=dict(l=40, r=20, t=50, b=40),
        plot_bgcolor='rgba(248,249,250,1)',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(contributions, top_n=8):
    """
    Render radar/spider chart showing top feature deviations.
    
    WHY RADAR CHART?
    - Shows "system health shape" at a glance
    - Multiple features compared simultaneously
    - Anomalous features visually "spike out" from the center
    - Normal system has a small, round shape
    - Anomalous system has a spiky, irregular shape
    """
    if not contributions:
        st.info("No feature data available")
        return
    
    # Take top N features by z-score
    top_features = contributions[:top_n]
    
    categories = [f['feature_display'] for f in top_features]
    values = [min(f['z_score'], 5) for f in top_features]  # Cap at 5 for display
    
    # Close the radar (first point = last point)
    categories.append(categories[0])
    values.append(values[0])
    
    # Normal reference (z-score = 1 is normal variation)
    normal_values = [1.0] * len(categories)
    
    fig = go.Figure()
    
    # Normal range
    fig.add_trace(go.Scatterpolar(
        r=normal_values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(64,192,87,0.15)',
        line=dict(color='rgba(64,192,87,0.5)', dash='dash'),
        name='Normal Range'
    ))
    
    # Current values
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(255,75,75,0.2)',
        line=dict(color='#FF4B4B', width=2),
        name='Current Reading',
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10))
        ),
        title=dict(text='🎯 System Health Radar', font=dict(size=16)),
        showlegend=True,
        legend=dict(x=0.0, y=-0.15, orientation='h'),
        height=400,
        margin=dict(l=40, r=40, t=50, b=60)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_feature_histogram(current_value, feature_name, history):
    """
    Render histogram showing where current value falls in distribution.
    
    WHY HISTOGRAM?
    - Visually proves WHY a value is anomalous
    - Shows the full distribution from training/history
    - Red line shows current value's position
    - If current value is in the tail → clearly anomalous
    """
    if not history or len(history) < 5:
        st.info("Need more data points for distribution chart")
        return
    
    # Get historical values for this feature
    hist_values = []
    for h in history:
        features = h['event']['features']
        if feature_name in features:
            hist_values.append(features[feature_name])
    
    if not hist_values:
        return
    
    fig = go.Figure()
    
    # Historical distribution
    fig.add_trace(go.Histogram(
        x=hist_values,
        name='Historical Values',
        marker_color='rgba(52,152,219,0.6)',
        nbinsx=20
    ))
    
    # Current value marker
    fig.add_vline(
        x=current_value,
        line_width=3,
        line_dash="solid",
        line_color="red",
        annotation_text=f"Current: {current_value:.2f}",
        annotation_position="top",
        annotation_font_color="red"
    )
    
    display_name = get_display_name(feature_name)
    fig.update_layout(
        title=dict(text=f'📊 {display_name} Distribution', font=dict(size=14)),
        xaxis_title=display_name,
        yaxis_title='Frequency',
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
        plot_bgcolor='rgba(248,249,250,1)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_key_metrics_cards(event_data):
    """
    Render raw metric values as cards (API Latency, FPS, Memory, etc.)
    
    WHY: Users want to see actual values, not just anomaly scores
    """
    raw_display = event_data.get('raw_display', {})
    
    if not raw_display:
        return
    
    cols = st.columns(len(raw_display))
    
    metric_icons = {
        'api_latency_ms': '⏱️',
        'ui_response_ms': '📱',
        'fps': '🖥️',
        'memory_mb': '💾'
    }
    
    for col, (feat, raw_val) in zip(cols, raw_display.items()):
        with col:
            icon = metric_icons.get(feat, '📊')
            display_name = get_display_name(feat)
            mapping = RAW_VALUE_MAPPING.get(feat, {})
            unit = mapping.get('unit', '')
            higher_bad = mapping.get('higher_is_bad', True)
            mean = mapping.get('mean', 0)
            
            # Color based on whether value is good or bad
            if higher_bad:
                is_bad = raw_val > mean * 1.5
            else:
                is_bad = raw_val < mean * 0.7
            
            color = '#FF4B4B' if is_bad else '#40C057'
            
            st.metric(
                label=f"{icon} {display_name}",
                value=f"{raw_val:.0f} {unit}",
                delta=f"{'⬆️ High' if is_bad and higher_bad else '⬇️ Low' if is_bad else '✅ Normal'}",
                delta_color="inverse" if (is_bad and higher_bad) or (is_bad and not higher_bad) else "normal"
            )


def render_data_table(history, max_rows=20):
    """
    Render table of recent events with predictions.
    
    WHY DATA TABLE?
    - Engineers want to see raw data
    - Easy to spot patterns across events
    - Can sort/filter to find specific anomalies
    """
    if not history:
        st.info("No events generated yet")
        return
    
    recent = history[-max_rows:]
    
    table_data = []
    for i, h in enumerate(recent):
        pred = h['prediction']
        event = h['event']
        
        row = {
            '#': len(history) - max_rows + i + 1,
            'Time': event['timestamp'],
            'Status': '🔴 Anomaly' if pred['is_anomaly'] else '🟢 Normal',
            'Score': f"{pred['anomaly_score_pct']:.1f}%",
            'Severity': f"{get_severity_icon(pred['severity'])} {pred['severity']}",
            'Top Trigger': pred['top_trigger'],
            'Type': event.get('anomaly_type', 'normal'),
        }
        
        # Add key raw values
        for feat in ['api_latency_ms', 'fps', 'memory_mb']:
            if feat in event.get('raw_display', {}):
                display = get_display_name(feat)
                row[display] = f"{event['raw_display'][feat]:.0f}"
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    st.dataframe(
        df,
        use_container_width=True,
        height=min(400, len(df) * 40 + 40),
        hide_index=True
    )


def render_batch_summary(history):
    """Render summary stats for batch analysis."""
    if not history:
        return
    
    total = len(history)
    anomalies = sum(1 for h in history if h['prediction']['is_anomaly'])
    normal = total - anomalies
    
    severity_counts = {}
    for h in history:
        sev = h['prediction']['severity']
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Events", total)
    col2.metric("🟢 Normal", normal)
    col3.metric("🔴 Anomalies", anomalies)
    col4.metric("Anomaly Rate", f"{anomalies/total*100:.1f}%")
    
    # Severity bar chart
    if anomalies > 0:
        severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL']
        severity_colors = ['#FF0000', '#FF6600', '#FFD700', '#00CC00', '#808080']
        
        fig = go.Figure(data=[
            go.Bar(
                x=[s for s in severity_order if severity_counts.get(s, 0) > 0],
                y=[severity_counts.get(s, 0) for s in severity_order if severity_counts.get(s, 0) > 0],
                marker_color=[c for s, c in zip(severity_order, severity_colors) 
                             if severity_counts.get(s, 0) > 0],
                text=[severity_counts.get(s, 0) for s in severity_order 
                      if severity_counts.get(s, 0) > 0],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Severity Distribution',
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor='rgba(248,249,250,1)'
        )
        
        st.plotly_chart(fig, use_container_width=True)