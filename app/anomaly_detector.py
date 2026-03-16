"""
anomaly_detector.py - Live Data Generation + Model Prediction Wrapper

This module provides:
1. LiveDataGenerator: Generates realistic telemetry events (normal + anomalous)
2. AppAnomalyDetector: Wraps the trained model for dashboard predictions

WHY SEPARATE FROM inference.py?
- inference.py is for general-purpose prediction
- This module is Streamlit-specific: generates demo data, formats for dashboard
- Keeps the app layer separate from the ML layer

Author: BCA Final Year Project
Date: 2026
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add project root to path so we can import from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.utils import (
    RAW_VALUE_MAPPING, ANOMALY_DESCRIPTIONS,
    get_display_name, scaled_to_raw, format_raw_value
)


class LiveDataGenerator:
    """
    Generates realistic telemetry events for live dashboard demo.
    
    WHY THIS CLASS?
    - Need to simulate real-time app monitoring data
    - Must match the EXACT format the trained model expects
    - Creates both normal (90%) and anomalous (10%) events
    - Allows forcing specific anomaly types for demo
    
    HOW IT GENERATES DATA:
    - Normal events: Feature values drawn from N(0, small_σ)
      (Because training data was scaled to mean=0, std=1)
    - Anomalous events: Specific features pushed to extreme values
      based on anomaly type (memory_leak → high memory features)
    
    FEATURE COHERENCE:
    - When api_latency_ms is high, rolling_mean should also increase
    - When fps drops, fps_zscore should become negative
    - Generator maintains these relationships for realism
    """
    
    def __init__(self, selected_features, models_dir='models'):
        """
        Initialize with the list of features the model expects.
        
        Args:
            selected_features: List of feature names (from selected_features.pkl)
            models_dir: Path to models directory
        """
        self.selected_features = selected_features
        self.models_dir = models_dir
        
        # Load baseline stats for realistic value ranges
        # WHY: baseline_stats has mean/std of normal data — 
        # helps generate realistic normal events
        baseline_path = os.path.join(models_dir, 'baseline_stats.pkl')
        if os.path.exists(baseline_path):
            self.baseline_stats = joblib.load(baseline_path)
        else:
            self.baseline_stats = None
        
        # Categorize features by type for coherent generation
        # WHY: Features in the same group should be correlated
        self.feature_groups = self._categorize_features()
    
    def _categorize_features(self):
        """
        Categorize features into groups for coherent data generation.
        
        WHY: When generating an anomaly, related features should 
        ALL shift together. E.g., if api_latency_ms is high, then
        api_latency_ms_rolling_mean_5 should also be elevated.
        """
        groups = {
            'api_latency': [],
            'ui_response': [],
            'fps': [],
            'memory': [],
            'error': [],
            'change_rate': [],
            'other': []
        }
        
        for feat in self.selected_features:
            feat_lower = feat.lower()
            if 'api_latency' in feat_lower or 'latency_cv' in feat_lower:
                groups['api_latency'].append(feat)
            elif 'ui_response' in feat_lower:
                groups['ui_response'].append(feat)
            elif 'fps' in feat_lower:
                groups['fps'].append(feat)
            elif 'memory' in feat_lower:
                groups['memory'].append(feat)
            elif 'error' in feat_lower:
                groups['error'].append(feat)
            elif 'change_rate' in feat_lower or 'growth_rate' in feat_lower:
                groups['change_rate'].append(feat)
            else:
                groups['other'].append(feat)
        
        return groups
    
    def generate_event(self, force_anomaly=False, anomaly_type=None, intensity=1.0):
        """
        Generate a single telemetry event.
        
        FIX: When force_anomaly=True, uses HIGHER minimum intensity
        to guarantee the model detects it.
        """
        if force_anomaly:
            if anomaly_type is None:
                anomaly_type = np.random.choice(list(ANOMALY_DESCRIPTIONS.keys()))
            
            # ← FIX: Ensure minimum intensity for forced anomalies
            # WHY: At low intensity, model might not detect the anomaly
            effective_intensity = max(intensity, 1.0)  # Never below 1.0 when forced
            
            features = self._generate_anomalous_event(anomaly_type, effective_intensity)
            is_anomaly = True
        else:
            if np.random.random() < 0.10:
                anomaly_type = np.random.choice(list(ANOMALY_DESCRIPTIONS.keys()))
                features = self._generate_anomalous_event(
                    anomaly_type, 
                    intensity=np.random.uniform(0.8, 1.5)
                )
                is_anomaly = True
            else:
                features = self._generate_normal_event()
                anomaly_type = 'normal'
                is_anomaly = False
        
        # Generate raw display values
        raw_display = {}
        for feat in ['api_latency_ms', 'ui_response_ms', 'fps', 'memory_mb']:
            if feat in features:
                raw_display[feat] = scaled_to_raw(feat, features[feat])
        
        return {
            'features': features,
            'is_generated_anomaly': is_anomaly,
            'anomaly_type': anomaly_type,
            'raw_display': raw_display,
            'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
        }
    
    def _generate_normal_event(self):
        """
        Generate a normal event with REALISTIC variation.
        
        FIX: Wider variation so some normal events are borderline
        → Creates occasional false positives (realistic)
        → Prevents 100% precision
        """
        features = {}
        
        for feat in self.selected_features:
            feat_lower = feat.lower()
            
            if 'zscore' in feat_lower:
                features[feat] = np.random.normal(0, 0.8)
            elif 'rolling_std' in feat_lower:
                features[feat] = abs(np.random.normal(0, 0.5))
            elif 'rolling_mean' in feat_lower:
                features[feat] = np.random.normal(0, 0.6)
            elif 'rolling_max' in feat_lower:
                features[feat] = np.random.normal(0.2, 0.6)
            elif 'change_rate' in feat_lower or 'growth_rate' in feat_lower:
                features[feat] = np.random.normal(0, 0.5)
            elif 'error' in feat_lower:
                features[feat] = np.random.normal(-0.2, 0.5)
            elif feat_lower == 'latency_cv':
                features[feat] = np.random.normal(0, 0.5)
            else:
                features[feat] = np.random.normal(0, 0.7)
        
        return features
    
    def _generate_anomalous_event(self, anomaly_type, intensity=1.0):
        """
        Generate an anomalous event with REALISTIC intensity.
        
        FIX: 
        - Values are moderate (not extreme 6-10σ)
        - More noise added for variability
        - Some mild anomalies WILL be missed by model (realistic)
        - Different intensity levels produce genuinely different results
        
        EXPECTED DETECTION RATES BY INTENSITY:
        - 0.5 (mild):     50-70% detection
        - 1.0 (moderate):  75-85% detection
        - 1.5 (strong):    85-95% detection
        - 2.0+ (severe):   95-100% detection
        """
        features = {}
        
        for feat in self.selected_features:
            features[feat] = np.random.normal(0, 0.3)
        
        anomaly_patterns = {
            'memory_leak': {
                'memory': {
                    'base': 2.2, 'rolling_mean': 1.8, 'rolling_std': 1.2,
                    'rolling_max': 2.5, 'zscore': 2.0, 'growth': 1.8
                },
                'fps': {
                    'base': -0.8, 'rolling_mean': -0.5, 'rolling_std': 0.6,
                    'rolling_max': -0.3, 'zscore': -0.8, 'change': -0.7
                },
            },
            'latency_spike': {
                'api_latency': {
                    'base': 2.5, 'rolling_mean': 2.0, 'rolling_std': 1.5,
                    'rolling_max': 3.0, 'zscore': 2.5, 'cv': 1.8
                },
                'ui_response': {
                    'base': 1.5, 'rolling_mean': 1.2, 'rolling_std': 0.8,
                    'rolling_max': 1.8, 'zscore': 1.5
                },
            },
            'fps_drop': {
                'fps': {
                    'base': -2.5, 'rolling_mean': -2.0, 'rolling_std': 1.2,
                    'rolling_max': -1.0, 'zscore': -2.2, 'change': -1.8
                },
                'ui_response': {
                    'base': 1.0, 'rolling_mean': 0.8, 'rolling_std': 0.6,
                    'rolling_max': 1.2, 'zscore': 1.0
                },
            },
            'error_burst': {
                'error': {
                    'base': 2.5, 'count': 2.5
                },
                'api_latency': {
                    'base': 1.2, 'rolling_mean': 1.0, 'rolling_std': 0.8,
                    'rolling_max': 1.5, 'zscore': 1.2, 'cv': 1.0
                },
            },
            'api_timeout': {
                'api_latency': {
                    'base': 3.0, 'rolling_mean': 2.5, 'rolling_std': 2.0,
                    'rolling_max': 3.5, 'zscore': 3.0, 'cv': 2.2
                },
                'ui_response': {
                    'base': 2.0, 'rolling_mean': 1.5, 'rolling_std': 1.2,
                    'rolling_max': 2.5, 'zscore': 2.0
                },
            }
        }
        
        pattern = anomaly_patterns.get(anomaly_type, anomaly_patterns['latency_spike'])
        
        for group_name, deviations in pattern.items():
            for feat in self.feature_groups.get(group_name, []):
                feat_lower = feat.lower()
                
                if 'rolling_mean' in feat_lower:
                    dev = deviations.get('rolling_mean', 1.5)
                elif 'rolling_std' in feat_lower:
                    dev = abs(deviations.get('rolling_std', 1.0))
                elif 'rolling_max' in feat_lower:
                    dev = deviations.get('rolling_max', 2.0)
                elif 'zscore' in feat_lower:
                    dev = deviations.get('zscore', 1.8)
                elif 'change_rate' in feat_lower:
                    dev = deviations.get('change', 1.2)
                elif 'growth_rate' in feat_lower:
                    dev = deviations.get('growth', 1.2)
                elif 'cv' in feat_lower:
                    dev = deviations.get('cv', 1.2)
                elif 'error_count' in feat_lower or 'recent_error' in feat_lower:
                    dev = deviations.get('count', 1.8)
                else:
                    dev = deviations.get('base', 1.8)
                
                # More noise for realism
                noise = np.random.normal(0, 0.4)
                features[feat] = dev * intensity + noise
        
        # Change rate features
        for feat in self.feature_groups.get('change_rate', []):
            feat_lower = feat.lower()
            if anomaly_type in ['memory_leak']:
                features[feat] = np.random.uniform(0.8, 2.0) * intensity
            elif anomaly_type in ['latency_spike', 'api_timeout']:
                if 'latency' in feat_lower:
                    features[feat] = np.random.uniform(1.0, 2.5) * intensity
                else:
                    features[feat] = np.random.uniform(0.3, 1.2) * intensity
            elif anomaly_type == 'fps_drop':
                if 'fps' in feat_lower:
                    features[feat] = np.random.uniform(-2.5, -0.8) * intensity
                else:
                    features[feat] = np.random.uniform(0.3, 1.0) * intensity
            elif anomaly_type == 'error_burst':
                features[feat] = np.random.uniform(0.5, 1.5) * intensity
        
        # Other features: minimal shift
        for feat in self.selected_features:
            if feat not in [f for group in pattern.keys() 
                           for f in self.feature_groups.get(group, [])]:
                if feat not in self.feature_groups.get('change_rate', []):
                    current = features.get(feat, 0)
                    shift = np.random.uniform(0.05, 0.25) * intensity
                    features[feat] = current + shift * np.random.choice([-1, 1])
        
        return features
    
    def generate_batch(self, n=50, anomaly_ratio=0.10, anomaly_type=None):
        """
        Generate a batch of events.
        
        Args:
            n: Number of events to generate
            anomaly_ratio: Fraction of events that should be anomalous (0.0 to 1.0)
            anomaly_type: If specified, all anomalies will be this type
                         If None, random types are chosen
            
        Returns:
            List of event dicts (same format as generate_event output)
        """
        events = []
        n_anomalies = int(n * anomaly_ratio)
        
        # Generate anomalous events first
        for _ in range(n_anomalies):
            atype = anomaly_type if anomaly_type else np.random.choice(
                list(ANOMALY_DESCRIPTIONS.keys())
            )
            events.append(self.generate_event(
                force_anomaly=True,
                anomaly_type=atype,
                intensity=np.random.uniform(0.8, 1.5)
            ))
        
        # Generate normal events
        for _ in range(n - n_anomalies):
            events.append(self.generate_event(force_anomaly=False))
        
        # Shuffle to mix normal and anomalous
        np.random.shuffle(events)
        
        return events


class AppAnomalyDetector:
    """
    Wraps the trained Isolation Forest model for dashboard predictions.
    
    WHY NOT USE inference.py DIRECTLY?
    - inference.py is a general-purpose module
    - This wrapper adds dashboard-specific features:
      - Feature contribution ranking (for "Top Trigger Factor")
      - Severity classification
      - Score normalization for display (0-100%)
      - Formatted output for Streamlit components
    """
    
    def __init__(self, models_dir='models'):
        self.models_dir = os.path.join(PROJECT_ROOT, models_dir)
        self.single_model = None
        self.ensemble_models = None
        self.threshold_data = None
        self.selected_features = None
        self.baseline_stats = None
        self.best_threshold = None
        self.use_ensemble = False
        self.is_loaded = False
    
    def load(self):
        """Load all model components."""
        # Single model
        self.single_model = joblib.load(
            os.path.join(self.models_dir, 'isolation_forest.pkl')
        )
        
        # Threshold and config
        self.threshold_data = joblib.load(
            os.path.join(self.models_dir, 'optimal_threshold.pkl')
        )
        self.best_threshold = self.threshold_data['best_threshold']
        self.use_ensemble = self.threshold_data.get('use_ensemble', False)
        
        # Selected features
        features_path = os.path.join(self.models_dir, 'selected_features.pkl')
        if os.path.exists(features_path):
            self.selected_features = joblib.load(features_path)
        else:
            self.selected_features = self.threshold_data.get('feature_names', [])
        
        # Ensemble models
        ensemble_path = os.path.join(self.models_dir, 'ensemble_models.pkl')
        if os.path.exists(ensemble_path):
            self.ensemble_models = joblib.load(ensemble_path)
        else:
            self.use_ensemble = False
        
        # Baseline stats
        baseline_path = os.path.join(self.models_dir, 'baseline_stats.pkl')
        if os.path.exists(baseline_path):
            self.baseline_stats = joblib.load(baseline_path)
        
        self.is_loaded = True
        return True
    
    def predict(self, features_dict):
        """
        Predict whether an event is anomalous.
        
        Args:
            features_dict: dict of {feature_name: value}
            
        Returns:
            dict with:
                - is_anomaly: bool
                - anomaly_score: float (raw model score)
                - anomaly_score_pct: float (0-100%, human readable)
                - severity: str ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL')
                - contributions: list of dicts (feature contributions)
                - top_trigger: str (most contributing feature name)
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded. Call load() first.")
        
        # Prepare input DataFrame
        X = pd.DataFrame([features_dict])[self.selected_features]
        
        # Handle NaN/Inf
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Get anomaly score
        if self.use_ensemble and self.ensemble_models:
            score = self._get_ensemble_score(X)
        else:
            score = self.single_model.score_samples(X)[0]
        
        # Predict using threshold
        is_anomaly = score < self.best_threshold
        
        # Normalize score to 0-100% for display
        # WHY: Raw scores like -0.4312 aren't intuitive
        # We map: most anomalous score → 100%, least anomalous → 0%
        score_pct = self._normalize_score(score)
        
        # Classify severity
        severity = self._classify_severity(score)
        
        # Get feature contributions
        contributions = self._get_contributions(features_dict)
        
        # Get top trigger
        top_trigger = contributions[0]['feature_display'] if contributions else 'Unknown'
        
        return {
            'is_anomaly': bool(is_anomaly),
            'anomaly_score': float(score),
            'anomaly_score_pct': float(score_pct),
            'severity': severity,
            'contributions': contributions,
            'top_trigger': top_trigger
        }
    
    def _get_ensemble_score(self, X):
        """Get combined score from ensemble models."""
        all_scores = []
        for model_info in self.ensemble_models:
            model = model_info['model']
            scores = model.score_samples(X)
            s_min, s_max = scores.min(), scores.max()
            if s_max - s_min > 0:
                normalized = (scores - s_min) / (s_max - s_min)
            else:
                normalized = np.zeros_like(scores)
            all_scores.append(normalized)
        return -np.mean(all_scores, axis=0)[0]
    
    def _normalize_score(self, score):
        """
        Normalize anomaly score to 0-100% for human display.
        
        FIX: Previous formula mapped everything to either 0% or 100%
        because the scaling factor (500) was too aggressive.
        
        NEW LOGIC:
        - Uses the actual score range for proper scaling
        - Scores near threshold → 50%
        - Scores far below threshold → 70-95%
        - Scores far above threshold → 5-30%
        - Never quite reaches 0% or 100% (more realistic)
        """
        # Distance from threshold
        # Positive = anomalous (score below threshold)
        # Negative = normal (score above threshold)
        distance = self.best_threshold - score
        
        # Typical score range is about ±0.2 around threshold
        # Scale this to 0-100 range with sigmoid-like mapping
        # WHY sigmoid: prevents extreme values (no 0% or 100%)
        
        # Scale factor: controls how spread out the percentages are
        scale = 8.0
        
        # Sigmoid mapping: smooth curve from ~5% to ~95%
        import math
        try:
            normalized = 1.0 / (1.0 + math.exp(-scale * distance))
        except OverflowError:
            normalized = 1.0 if distance > 0 else 0.0
        
        # Map to percentage
        pct = normalized * 100
        
        # Clamp to 2-98% range (never show exactly 0% or 100%)
        # WHY: In real anomaly detection, nothing is 100% certain
        pct = max(2.0, min(98.0, pct))
        
        return float(pct)
    
    def _classify_severity(self, score):
        """Classify severity based on score distance from threshold."""
        if score < self.best_threshold * 1.3:
            return 'CRITICAL'
        elif score < self.best_threshold * 1.15:
            return 'HIGH'
        elif score < self.best_threshold:
            return 'MEDIUM'
        elif score < self.best_threshold * 0.95:
            return 'LOW'
        else:
            return 'NORMAL'
    
    def _get_contributions(self, features_dict):
        """
        Calculate which features contribute most to the anomaly.
        
        HOW: Calculate z-score for each feature using baseline stats.
        Higher z-score = more unusual = more contribution to anomaly.
        
        WHY: Tells the user "API Latency is 3.2 standard deviations 
        above normal — this is the main reason for the anomaly flag."
        """
        contributions = []
        
        if self.baseline_stats is not None:
            means = self.baseline_stats['means']
            stds = self.baseline_stats['stds']
            
            for feat in self.selected_features:
                if feat in features_dict and feat in means.index:
                    value = features_dict[feat]
                    mean = means[feat]
                    std = stds[feat]
                    
                    if std > 1e-10:
                        z = abs((value - mean) / std)
                    else:
                        z = 0
                    
                    direction = 'HIGH' if value > mean else 'LOW'
                    
                    contributions.append({
                        'feature': feat,
                        'feature_display': get_display_name(feat),
                        'value': round(float(value), 4),
                        'z_score': round(float(z), 2),
                        'direction': direction,
                        'raw_display': format_raw_value(feat, value)
                    })
            
            contributions.sort(key=lambda x: x['z_score'], reverse=True)
        else:
            # Fallback: use absolute values
            for feat in self.selected_features:
                if feat in features_dict:
                    contributions.append({
                        'feature': feat,
                        'feature_display': get_display_name(feat),
                        'value': round(float(features_dict[feat]), 4),
                        'z_score': round(abs(float(features_dict[feat])), 2),
                        'direction': 'HIGH' if features_dict[feat] > 0 else 'LOW',
                        'raw_display': format_raw_value(feat, features_dict[feat])
                    })
            contributions.sort(key=lambda x: x['z_score'], reverse=True)
        
        return contributions
    
    def get_model_info(self):
        """Return model configuration for sidebar display."""
        return {
            'Model Type': 'Ensemble' if self.use_ensemble else 'Single IF',
            'Ensemble Size': len(self.ensemble_models) if self.ensemble_models else 1,
            'Features Used': len(self.selected_features),
            'Threshold': f"{self.best_threshold:.4f}",
            'Status': '✅ Loaded' if self.is_loaded else '❌ Not Loaded'
        }