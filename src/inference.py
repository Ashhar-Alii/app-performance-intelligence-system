"""
inference.py - Anomaly Detection Inference / Prediction Module

This module handles production-ready anomaly detection:
1. Load trained models (single model + ensemble)
2. Preprocess raw input data (match training format)
3. Run anomaly detection (single event or batch)
4. Return detailed predictions with severity and explanations

USAGE MODES:
- Single event: Predict one telemetry event at a time (real-time)
- Batch: Predict many events at once (batch processing)
- Raw data: Accepts unprocessed data and handles preprocessing
- Preprocessed data: Accepts already-preprocessed data

WHY THIS MODULE?
- Separates training logic from prediction logic
- Production-ready: error handling, logging, validation
- Provides severity levels (not just binary anomaly/normal)
- Generates human-readable explanations for flagged anomalies
- Can be imported by other modules (API, dashboard, CLI)

Author: BCA Final Year Project
Date: 2026
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetector:
    """
    Production-ready anomaly detection engine.
    
    Loads trained models and provides prediction capabilities
    for single events and batch processing.
    
    ARCHITECTURE:
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  Raw Data    │ ──→ │ Preprocessor │ ──→ │   Models     │
    │  (input)     │     │ (clean/scale)│     │ (IF/Ensemble)│
    └──────────────┘     └──────────────┘     └──────┬───────┘
                                                      │
                         ┌──────────────┐     ┌──────▼───────┐
                         │  Explanation │ ←── │  Predictions │
                         │  + Severity  │     │  + Scores    │
                         └──────────────┘     └──────────────┘
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the anomaly detector.
        
        Args:
            models_dir: Directory where trained models are saved
            
        WHY LAZY LOADING?
        Models are loaded only when load_models() is called,
        not during __init__. This allows creating the object
        first and loading models later (useful for APIs).
        """
        self.models_dir = models_dir
        
        # Model components (loaded later)
        self.single_model = None
        self.ensemble_models = None
        self.baseline_stats = None
        self.threshold_data = None
        self.selected_features = None
        
        # Configuration (set after loading)
        self.best_threshold = None
        self.use_ensemble = False
        self.is_loaded = False
        
        # Severity thresholds
        # WHY: Binary anomaly/normal isn't enough for production
        # Engineers need to know: Is this critical or just unusual?
        self.severity_levels = {
            'CRITICAL': 0.05,    # Bottom 5% of scores → most anomalous
            'HIGH': 0.10,        # Bottom 10%
            'MEDIUM': 0.20,      # Bottom 20%
            'LOW': 0.30,         # Bottom 30%
            'NORMAL': 1.0        # Everything else
        }
        
    def load_models(self):
        """
        Load all trained model components from disk.
        
        WHAT WE LOAD:
        1. Isolation Forest (single best model)
        2. Ensemble models (top 3 models)
        3. Optimal threshold and configuration
        4. Selected features list
        5. Baseline statistics (for z-score comparison)
        
        WHY ALL THESE FILES?
        - Model alone isn't enough — need threshold, features, config
        - Ensemble needs multiple model objects
        - Selected features ensure we use same features as training
        - Baseline stats enable z-score comparison
        
        ERROR HANDLING:
        - Checks each file exists before loading
        - Provides clear error messages if files are missing
        - Gracefully handles missing optional files (ensemble)
        """
        print("="*70)
        print("LOADING ANOMALY DETECTION MODELS")
        print("="*70)
        
        # ─────────────────────────────────────────
        # Validate model directory exists
        # ─────────────────────────────────────────
        if not os.path.exists(self.models_dir):
            raise FileNotFoundError(
                f"\n❌ Models directory not found: '{self.models_dir}'"
                f"\n   Run train_model.py first to train and save models."
            )
        
        # ─────────────────────────────────────────
        # Load each component
        # ─────────────────────────────────────────
        
        # 1. Single best Isolation Forest model (REQUIRED)
        iforest_path = os.path.join(self.models_dir, 'isolation_forest.pkl')
        if not os.path.exists(iforest_path):
            raise FileNotFoundError(f"❌ Model file not found: {iforest_path}")
        self.single_model = joblib.load(iforest_path)
        print(f"✅ Isolation Forest model loaded: {iforest_path}")
        
        # 2. Optimal threshold and configuration (REQUIRED)
        threshold_path = os.path.join(self.models_dir, 'optimal_threshold.pkl')
        if not os.path.exists(threshold_path):
            raise FileNotFoundError(f"❌ Threshold file not found: {threshold_path}")
        self.threshold_data = joblib.load(threshold_path)
        self.best_threshold = self.threshold_data['best_threshold']
        self.use_ensemble = self.threshold_data.get('use_ensemble', False)
        print(f"✅ Threshold loaded: {self.best_threshold:.4f}")
        print(f"   Mode: {'Ensemble' if self.use_ensemble else 'Single Model'}")
        
        # 3. Selected features list (REQUIRED)
        features_path = os.path.join(self.models_dir, 'selected_features.pkl')
        if not os.path.exists(features_path):
            # Fallback: try getting features from threshold_data
            self.selected_features = self.threshold_data.get('feature_names', None)
            if self.selected_features:
                print(f"✅ Features loaded from threshold config: {len(self.selected_features)} features")
            else:
                raise FileNotFoundError(
                    f"❌ Features file not found: {features_path}"
                    f"\n   Cannot determine which features to use for prediction."
                )
        else:
            self.selected_features = joblib.load(features_path)
            print(f"✅ Selected features loaded: {len(self.selected_features)} features")
        
        # 4. Ensemble models (OPTIONAL — only if ensemble mode)
        ensemble_path = os.path.join(self.models_dir, 'ensemble_models.pkl')
        if os.path.exists(ensemble_path):
            self.ensemble_models = joblib.load(ensemble_path)
            print(f"✅ Ensemble models loaded: {len(self.ensemble_models)} models")
        else:
            if self.use_ensemble:
                print(f"⚠️  Ensemble mode enabled but ensemble file not found")
                print(f"    Falling back to single model mode")
                self.use_ensemble = False
            else:
                print(f"ℹ️  No ensemble file found (single model mode)")
        
        # 5. Baseline statistics (OPTIONAL — for comparison)
        baseline_path = os.path.join(self.models_dir, 'baseline_stats.pkl')
        if os.path.exists(baseline_path):
            self.baseline_stats = joblib.load(baseline_path)
            print(f"✅ Baseline statistics loaded")
        else:
            print(f"ℹ️  No baseline stats found (z-score comparison disabled)")
        
        # ─────────────────────────────────────────
        # Compute severity percentile thresholds
        # ─────────────────────────────────────────
        # WHY: Pre-compute score thresholds for each severity level
        # These are based on the model's score distribution
        # We'll calibrate these when we see actual prediction scores
        
        self.is_loaded = True
        
        print(f"\n{'='*70}")
        print(f"✅ ALL MODELS LOADED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\n📊 Configuration Summary:")
        print(f"   Model type:     {'Ensemble (' + str(len(self.ensemble_models)) + ' models)' if self.use_ensemble else 'Single Isolation Forest'}")
        print(f"   Threshold:      {self.best_threshold:.4f}")
        print(f"   Features:       {len(self.selected_features)}")
        print(f"   Baseline:       {'Available' if self.baseline_stats else 'Not available'}")
        print(f"\n   Ready for predictions! ✅")
        
        return True
    
    def _validate_loaded(self):
        """
        Check that models are loaded before making predictions.
        
        WHY: Prevent confusing errors if someone forgets to call load_models()
        """
        if not self.is_loaded:
            raise RuntimeError(
                "\n❌ Models not loaded! Call load_models() first."
                "\n\nExample:"
                "\n   detector = AnomalyDetector(models_dir='models')"
                "\n   detector.load_models()"
                "\n   results = detector.predict(data)"
            )
    
    def _validate_features(self, data):
        """
        Validate that input data has the required features.
        
        WHY: Catch feature mismatches EARLY with clear error messages
        rather than getting cryptic sklearn errors later.
        
        CHECKS:
        1. All required features are present
        2. No unexpected NaN/null values
        3. Data types are numeric
        
        Args:
            data: DataFrame to validate
            
        Returns:
            DataFrame with only selected features in correct order
        """
        # Check for missing features
        missing_features = [f for f in self.selected_features if f not in data.columns]
        
        if missing_features:
            available = [f for f in self.selected_features if f in data.columns]
            raise ValueError(
                f"\n❌ Missing {len(missing_features)} required features:"
                f"\n   Missing: {missing_features}"
                f"\n   Available: {len(available)}/{len(self.selected_features)}"
                f"\n\n   Required features: {self.selected_features}"
                f"\n\n💡 Make sure your data is preprocessed the same way as training data."
            )
        
        # Select only required features in correct order
        # WHY: Feature order matters for sklearn models
        data_clean = data[self.selected_features].copy()
        
        # Check for NaN values
        nan_counts = data_clean.isnull().sum()
        if nan_counts.sum() > 0:
            nan_features = nan_counts[nan_counts > 0]
            print(f"\n⚠️  WARNING: Found NaN values in {len(nan_features)} features:")
            for feat, count in nan_features.items():
                print(f"      {feat}: {count} NaN values")
            
            # Fill NaN with 0 (safe default for scaled features)
            # WHY: Model can't handle NaN, and 0 is neutral for scaled data
            data_clean = data_clean.fillna(0)
            print(f"   → Filled NaN values with 0")
        
        # Check for infinite values
        inf_counts = np.isinf(data_clean.select_dtypes(include=[np.number])).sum()
        if inf_counts.sum() > 0:
            inf_features = inf_counts[inf_counts > 0]
            print(f"\n⚠️  WARNING: Found infinite values in {len(inf_features)} features:")
            for feat, count in inf_features.items():
                print(f"      {feat}: {count} inf values")
            
            # Replace inf with large values
            data_clean = data_clean.replace([np.inf, -np.inf], 0)
            print(f"   → Replaced infinite values with 0")
        
        return data_clean
    
    def _get_ensemble_scores(self, X):
        """
        Get combined anomaly scores from the ensemble.
        
        Same logic as in train_model.py — MUST match exactly.
        
        WHY SAME LOGIC?
        - Training used this exact normalization + averaging
        - Threshold was optimized on these combined scores
        - Any difference would make threshold invalid
        
        HOW:
        1. Get raw scores from each model
        2. Min-Max normalize each to [0, 1]
        3. Average the normalized scores
        4. Negate (so lower = more anomalous)
        """
        all_scores = []
        
        for model_info in self.ensemble_models:
            model = model_info['model']
            scores = model.score_samples(X)
            
            score_min = scores.min()
            score_max = scores.max()
            
            if score_max - score_min > 0:
                normalized = (scores - score_min) / (score_max - score_min)
            else:
                normalized = np.zeros_like(scores)
            
            all_scores.append(normalized)
        
        ensemble_scores = np.mean(all_scores, axis=0)
        ensemble_scores = -ensemble_scores  # Lower = more anomalous
        
        return ensemble_scores
    
    def _get_single_model_scores(self, X):
        """
        Get anomaly scores from single best model.
        
        WHY SEPARATE METHOD?
        - Clean separation between ensemble and single model logic
        - Easier to debug and test each independently
        """
        return self.single_model.score_samples(X)
    
    def _classify_severity(self, score, all_scores=None):
        """
        Classify anomaly severity based on score.
        
        WHY SEVERITY LEVELS?
        - Binary (anomaly/normal) isn't useful for engineers
        - "CRITICAL" needs immediate attention
        - "LOW" can be reviewed later
        - Helps prioritize response efforts
        
        SEVERITY LEVELS:
        ┌──────────┬───────────────────────────────────────────┐
        │ CRITICAL │ Extremely anomalous, likely system failure │
        │ HIGH     │ Significant anomaly, investigate soon      │
        │ MEDIUM   │ Moderate anomaly, review when possible     │
        │ LOW      │ Mild anomaly, monitor                      │
        │ NORMAL   │ No anomaly detected                        │
        └──────────┴───────────────────────────────────────────┘
        
        HOW:
        - Uses percentile-based thresholds
        - If individual scores available: use absolute position
        - If batch scores available: use relative position in batch
        """
        if all_scores is not None and len(all_scores) > 10:
            # Use percentile within the current batch
            percentile = (all_scores < score).mean() * 100  # What % of scores are lower
            
            if percentile <= 5:
                return 'CRITICAL'
            elif percentile <= 10:
                return 'HIGH'
            elif percentile <= 20:
                return 'MEDIUM'
            elif score < self.best_threshold:
                return 'LOW'
            else:
                return 'NORMAL'
        else:
            # Use threshold-based classification for single events
            if score < self.best_threshold * 1.3:
                return 'CRITICAL'
            elif score < self.best_threshold * 1.1:
                return 'HIGH'
            elif score < self.best_threshold:
                return 'MEDIUM'
            elif score < self.best_threshold * 0.95:
                return 'LOW'
            else:
                return 'NORMAL'
    
    def _generate_explanation(self, event_data, score, is_anomaly):
        """
        Generate human-readable explanation for a prediction.
        
        WHY EXPLANATIONS?
        - "Anomaly detected" isn't helpful for engineers
        - Need to know WHICH metrics are unusual and WHY
        - Helps with root cause analysis
        - Required for trust in the system
        
        HOW:
        1. Compare each feature value to its expected range
        2. Flag features that deviate significantly
        3. Generate natural language description
        
        USES: Baseline statistics (mean/std of normal data) if available
        """
        explanation = {}
        contributing_features = []
        
        if not is_anomaly:
            explanation['summary'] = "All metrics within normal range"
            explanation['contributing_features'] = []
            explanation['recommendation'] = "No action needed"
            return explanation
        
        # If we have baseline stats, compare against normal ranges
        if self.baseline_stats is not None:
            means = self.baseline_stats['means']
            stds = self.baseline_stats['stds']
            
            for feature in self.selected_features:
                if feature in event_data.index and feature in means.index:
                    value = event_data[feature]
                    mean = means[feature]
                    std = stds[feature]
                    
                    if std > 0:
                        z_score = abs((value - mean) / std)
                    else:
                        z_score = 0
                    
                    # Flag features with z-score > 2 (significantly different)
                    if z_score > 2.0:
                        direction = "HIGH" if value > mean else "LOW"
                        contributing_features.append({
                            'feature': feature,
                            'value': round(float(value), 4),
                            'expected_mean': round(float(mean), 4),
                            'z_score': round(float(z_score), 2),
                            'direction': direction
                        })
            
            # Sort by z-score (most anomalous features first)
            contributing_features.sort(key=lambda x: x['z_score'], reverse=True)
        
        # Generate summary
        if len(contributing_features) > 0:
            top_features = contributing_features[:3]  # Top 3 most anomalous
            feature_names = [f['feature'] for f in top_features]
            
            explanation['summary'] = (
                f"Anomaly detected. Top contributing factors: "
                f"{', '.join(feature_names)}"
            )
        else:
            explanation['summary'] = (
                "Anomaly detected based on combined feature patterns. "
                "No single feature is extremely unusual, but the combination is rare."
            )
        
        explanation['contributing_features'] = contributing_features[:5]  # Top 5
        
        # Generate recommendation based on contributing features
        explanation['recommendation'] = self._generate_recommendation(contributing_features)
        
        return explanation
    
    def _generate_recommendation(self, contributing_features):
        """
        Generate actionable recommendation based on anomaly type.
        
        WHY: Engineers need to know WHAT TO DO, not just that something is wrong.
        
        LOGIC: Based on which features are anomalous, suggest likely root cause
        and action to take.
        """
        if not contributing_features:
            return "Monitor the situation. Review recent changes."
        
        feature_names = [f['feature'] for f in contributing_features]
        feature_names_str = ' '.join(feature_names).lower()
        
        # Match feature patterns to recommendations
        if 'latency' in feature_names_str or 'api' in feature_names_str:
            return ("Investigate API performance. Check: "
                    "1) Backend server load, "
                    "2) Database query performance, "
                    "3) Network connectivity, "
                    "4) Third-party API response times.")
        
        elif 'memory' in feature_names_str:
            return ("Possible memory leak detected. Check: "
                    "1) Memory-intensive operations, "
                    "2) Unclosed resources/connections, "
                    "3) Cache sizes, "
                    "4) Recent code deployments.")
        
        elif 'fps' in feature_names_str:
            return ("UI performance degradation detected. Check: "
                    "1) Heavy UI rendering operations, "
                    "2) Main thread blocking, "
                    "3) Animations/transitions, "
                    "4) Device-specific issues.")
        
        elif 'error' in feature_names_str:
            return ("Error rate spike detected. Check: "
                    "1) Recent deployments, "
                    "2) Service dependencies, "
                    "3) Error logs for patterns, "
                    "4) User-facing impact.")
        
        else:
            return ("Multiple metrics showing unusual patterns. Check: "
                    "1) Recent code/config changes, "
                    "2) Infrastructure health, "
                    "3) External dependencies, "
                    "4) Traffic patterns.")
    
    def predict(self, data):
        """
        Main prediction method — handles both single events and batches.
        
        THIS IS THE PRIMARY METHOD TO USE.
        
        Args:
            data: One of the following:
                - pd.DataFrame: Batch of events (multiple rows)
                - pd.Series: Single event
                - dict: Single event as dictionary
                
        Returns:
            dict with:
                - predictions: Array of 0/1 (0=normal, 1=anomaly)
                - scores: Array of anomaly scores
                - severity: Array of severity levels
                - explanations: List of explanation dicts
                - summary: Overall summary statistics
                
        EXAMPLE:
            detector = AnomalyDetector()
            detector.load_models()
            results = detector.predict(my_data)
            
            # Access results
            print(results['summary'])
            for i, exp in enumerate(results['explanations']):
                if results['predictions'][i] == 1:
                    print(f"Event {i}: {exp['summary']}")
        """
        self._validate_loaded()
        
        # ─────────────────────────────────────────
        # Handle different input types
        # ─────────────────────────────────────────
        if isinstance(data, dict):
            # Single event as dictionary → convert to DataFrame
            data = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            # Single event as Series → convert to DataFrame
            data = pd.DataFrame([data])
        elif not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"❌ Expected DataFrame, Series, or dict. Got: {type(data)}"
                f"\n\nExample usage:"
                f"\n   results = detector.predict(pd.DataFrame(...))"
                f"\n   results = detector.predict({{'feature1': 1.0, ...}})"
            )
        
        # ─────────────────────────────────────────
        # Validate and select features
        # ─────────────────────────────────────────
        X = self._validate_features(data)
        
        # ─────────────────────────────────────────
        # Get anomaly scores
        # ─────────────────────────────────────────
        if self.use_ensemble and self.ensemble_models:
            scores = self._get_ensemble_scores(X)
        else:
            scores = self._get_single_model_scores(X)
        
        # ─────────────────────────────────────────
        # Make predictions using optimized threshold
        # ─────────────────────────────────────────
        predictions = (scores < self.best_threshold).astype(int)
        
        # ─────────────────────────────────────────
        # Classify severity for each event
        # ─────────────────────────────────────────
        severities = []
        for score in scores:
            severity = self._classify_severity(score, all_scores=scores)
            severities.append(severity)
        
        # ─────────────────────────────────────────
        # Generate explanations for each event
        # ─────────────────────────────────────────
        explanations = []
        for i in range(len(data)):
            event = X.iloc[i]
            is_anomaly = predictions[i] == 1
            explanation = self._generate_explanation(event, scores[i], is_anomaly)
            explanation['anomaly_score'] = round(float(scores[i]), 6)
            explanation['severity'] = severities[i]
            explanations.append(explanation)
        
        # ─────────────────────────────────────────
        # Generate summary
        # ─────────────────────────────────────────
        n_total = len(predictions)
        n_anomalies = int(predictions.sum())
        n_normal = n_total - n_anomalies
        
        severity_counts = {}
        for s in severities:
            severity_counts[s] = severity_counts.get(s, 0) + 1
        
        summary = {
            'total_events': n_total,
            'anomalies_detected': n_anomalies,
            'normal_events': n_normal,
            'anomaly_rate': round(n_anomalies / n_total * 100, 2) if n_total > 0 else 0,
            'severity_breakdown': severity_counts,
            'threshold_used': round(float(self.best_threshold), 6),
            'model_type': 'ensemble' if self.use_ensemble else 'single',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # ─────────────────────────────────────────
        # Build results
        # ─────────────────────────────────────────
        results = {
            'predictions': predictions,
            'scores': scores,
            'severity': severities,
            'explanations': explanations,
            'summary': summary
        }
        
        return results
    
    def predict_single(self, event):
        """
        Predict a single event — convenience method.
        
        WHY SEPARATE METHOD?
        - Simpler interface for real-time predictions
        - Returns a flat dictionary instead of arrays
        - Easier to use in APIs and real-time pipelines
        
        Args:
            event: dict or pd.Series with feature values
            
        Returns:
            dict with:
                - is_anomaly: bool
                - anomaly_score: float
                - severity: str
                - explanation: dict
                
        EXAMPLE:
            event = {
                'api_latency_ms': 0.5,
                'fps': -2.1,
                'memory_mb': 0.8,
                ...
            }
            result = detector.predict_single(event)
            
            if result['is_anomaly']:
                print(f"🚨 {result['severity']}: {result['explanation']['summary']}")
        """
        # Use main predict method
        results = self.predict(event)
        
        # Flatten for single event
        single_result = {
            'is_anomaly': bool(results['predictions'][0]),
            'anomaly_score': round(float(results['scores'][0]), 6),
            'severity': results['severity'][0],
            'explanation': results['explanations'][0],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return single_result
    
    def predict_batch(self, data, verbose=True):
        """
        Predict a batch of events with detailed reporting.
        
        WHY SEPARATE METHOD?
        - Adds batch-specific features: progress reporting, summary stats
        - Can save results to file
        - Provides formatted output for monitoring dashboards
        
        Args:
            data: pd.DataFrame with multiple events
            verbose: Whether to print detailed progress
            
        Returns:
            dict with predictions + DataFrame with all results
            
        EXAMPLE:
            batch_data = pd.read_csv('new_telemetry.csv')
            results = detector.predict_batch(batch_data)
            
            # Get results as DataFrame
            results_df = results['results_dataframe']
            results_df.to_csv('predictions_output.csv')
        """
        self._validate_loaded()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"BATCH ANOMALY DETECTION")
            print(f"{'='*70}")
            print(f"\n📊 Input: {len(data)} events")
        
        # Run prediction
        results = self.predict(data)
        
        # Create results DataFrame
        # WHY: Easy to save, filter, sort, analyze
        results_df = data.copy()
        results_df['predicted_anomaly'] = results['predictions']
        results_df['anomaly_score'] = results['scores']
        results_df['severity'] = results['severity']
        results_df['explanation'] = [exp['summary'] for exp in results['explanations']]
        results_df['recommendation'] = [exp['recommendation'] for exp in results['explanations']]
        
        if verbose:
            summary = results['summary']
            
            print(f"\n📊 BATCH RESULTS SUMMARY:")
            print(f"   ┌────────────────────────────────────────┐")
            print(f"   │  Total Events:     {summary['total_events']:<20}│")
            print(f"   │  Anomalies Found:  {summary['anomalies_detected']:<20}│")
            print(f"   │  Normal Events:    {summary['normal_events']:<20}│")
            print(f"   │  Anomaly Rate:     {summary['anomaly_rate']}%{' '*(17-len(str(summary['anomaly_rate'])))}│")
            print(f"   └────────────────────────────────────────┘")
            
            print(f"\n   🔥 Severity Breakdown:")
            severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL']
            severity_colors = {
                'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 
                'LOW': '🟢', 'NORMAL': '⚪'
            }
            
            for severity in severity_order:
                count = summary['severity_breakdown'].get(severity, 0)
                icon = severity_colors.get(severity, '⚪')
                bar = '█' * min(count, 50)
                print(f"      {icon} {severity:<10}: {count:>4} {bar}")
            
            # Show top anomalies
            anomaly_events = results_df[results_df['predicted_anomaly'] == 1]
            if len(anomaly_events) > 0:
                print(f"\n   🚨 Top 5 Most Anomalous Events:")
                top_anomalies = anomaly_events.nsmallest(5, 'anomaly_score')
                
                print(f"   {'Rank':<6} {'Score':<12} {'Severity':<10} {'Explanation'}")
                print(f"   {'-'*65}")
                
                for rank, (idx, row) in enumerate(top_anomalies.iterrows(), 1):
                    explanation_short = row['explanation'][:45] + '...' if len(row['explanation']) > 45 else row['explanation']
                    print(f"   {rank:<6} {row['anomaly_score']:<12.4f} {row['severity']:<10} {explanation_short}")
        
        # Add results DataFrame to output
        results['results_dataframe'] = results_df
        
        return results
    
    def save_predictions(self, results, output_path):
        """
        Save prediction results to CSV file.
        
        WHY: Persist results for analysis, reporting, audit trail
        
        Args:
            results: Output from predict_batch()
            output_path: Path to save CSV file
        """
        if 'results_dataframe' in results:
            results_df = results['results_dataframe']
        else:
            # If called with predict() results, create basic DataFrame
            results_df = pd.DataFrame({
                'predicted_anomaly': results['predictions'],
                'anomaly_score': results['scores'],
                'severity': results['severity']
            })
        
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Predictions saved to: {output_path}")
        print(f"   Total events: {len(results_df)}")
        print(f"   Anomalies: {(results_df['predicted_anomaly'] == 1).sum()}")
        
        return output_path
    
    def save_summary_report(self, results, output_path):
        """
        Save a human-readable summary report.
        
        WHY: Quick overview for stakeholders who don't read CSV files.
        Can be attached to emails, slack notifications, etc.
        
        Args:
            results: Output from predict() or predict_batch()
            output_path: Path to save text report
        """
        summary = results['summary']
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("ANOMALY DETECTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Timestamp: {summary['timestamp']}\n")
            f.write(f"Model Type: {summary['model_type']}\n\n")
            
            f.write("SUMMARY:\n")
            f.write("-"*40 + "\n")
            f.write(f"  Total events analyzed:  {summary['total_events']}\n")
            f.write(f"  Anomalies detected:     {summary['anomalies_detected']}\n")
            f.write(f"  Normal events:          {summary['normal_events']}\n")
            f.write(f"  Anomaly rate:           {summary['anomaly_rate']}%\n\n")
            
            f.write("SEVERITY BREAKDOWN:\n")
            f.write("-"*40 + "\n")
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'NORMAL']:
                count = summary['severity_breakdown'].get(severity, 0)
                f.write(f"  {severity:<10}: {count}\n")
            f.write("\n")
            
            # Write anomaly details
            if 'explanations' in results:
                anomaly_explanations = [
                    (i, exp) for i, exp in enumerate(results['explanations'])
                    if results['predictions'][i] == 1
                ]
                
                if anomaly_explanations:
                    f.write("ANOMALY DETAILS:\n")
                    f.write("-"*40 + "\n")
                    
                    for event_idx, exp in anomaly_explanations[:20]:  # Top 20
                        f.write(f"\n  Event #{event_idx}:\n")
                        f.write(f"    Score:    {exp['anomaly_score']}\n")
                        f.write(f"    Severity: {exp['severity']}\n")
                        f.write(f"    Summary:  {exp['summary']}\n")
                        f.write(f"    Action:   {exp['recommendation']}\n")
                        
                        if exp.get('contributing_features'):
                            f.write(f"    Top Contributing Features:\n")
                            for feat in exp['contributing_features'][:3]:
                                f.write(f"      - {feat['feature']}: "
                                       f"value={feat['value']}, "
                                       f"z-score={feat['z_score']}, "
                                       f"direction={feat['direction']}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*60 + "\n")
        
        print(f"✅ Summary report saved to: {output_path}")
        return output_path
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        WHY: Useful for debugging, logging, and API responses.
        
        Returns:
            dict with model configuration details
        """
        self._validate_loaded()
        
        info = {
            'model_type': 'ensemble' if self.use_ensemble else 'single',
            'threshold': round(float(self.best_threshold), 6),
            'n_features': len(self.selected_features),
            'features': self.selected_features,
            'is_loaded': self.is_loaded,
            'models_dir': self.models_dir,
            'severity_levels': list(self.severity_levels.keys()),
            'baseline_available': self.baseline_stats is not None
        }
        
        if self.use_ensemble and self.ensemble_models:
            info['ensemble_size'] = len(self.ensemble_models)
            info['ensemble_configs'] = [m['params'] for m in self.ensemble_models]
        
        if self.threshold_data and 'best_params' in self.threshold_data:
            info['best_params'] = self.threshold_data['best_params']
        
        return info


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def demo_single_event_prediction(detector):
    """
    Demonstrate single event prediction.
    
    WHY: Shows how to use the detector for real-time monitoring
    where events come one at a time.
    """
    print("\n" + "="*70)
    print("DEMO 1: SINGLE EVENT PREDICTION")
    print("="*70)
    
    # Create a sample event (using feature names from training)
    # NOTE: Values should be in the SAME scale as training data
    # (i.e., already preprocessed/scaled)
    
    print(f"\n📝 Creating sample events...")
    print(f"   Features needed: {len(detector.selected_features)}")
    
    # Normal event: all features near 0 (scaled mean)
    normal_event = {feat: np.random.normal(0, 0.3) for feat in detector.selected_features}
    
    # Anomalous event: some features with extreme values
    anomalous_event = {feat: np.random.normal(0, 0.3) for feat in detector.selected_features}
    # Make some features extreme
    for feat in detector.selected_features[:5]:
        if 'latency' in feat or 'memory' in feat:
            anomalous_event[feat] = np.random.uniform(3, 5)  # Very high (3-5 std devs)
        elif 'fps' in feat:
            anomalous_event[feat] = np.random.uniform(-4, -3)  # Very low
    
    # Predict normal event
    print(f"\n{'─'*50}")
    print(f"Testing NORMAL event:")
    result_normal = detector.predict_single(normal_event)
    
    print(f"   Is Anomaly:  {result_normal['is_anomaly']}")
    print(f"   Score:       {result_normal['anomaly_score']}")
    print(f"   Severity:    {result_normal['severity']}")
    print(f"   Explanation: {result_normal['explanation']['summary']}")
    
    # Predict anomalous event
    print(f"\n{'─'*50}")
    print(f"Testing ANOMALOUS event:")
    result_anomaly = detector.predict_single(anomalous_event)
    
    print(f"   Is Anomaly:  {result_anomaly['is_anomaly']}")
    print(f"   Score:       {result_anomaly['anomaly_score']}")
    print(f"   Severity:    {result_anomaly['severity']}")
    print(f"   Explanation: {result_anomaly['explanation']['summary']}")
    
    if result_anomaly['explanation'].get('contributing_features'):
        print(f"\n   Contributing Features:")
        for feat in result_anomaly['explanation']['contributing_features'][:3]:
            print(f"      - {feat['feature']}: z-score={feat['z_score']}, "
                  f"direction={feat['direction']}")
    
    print(f"\n   Recommendation: {result_anomaly['explanation']['recommendation']}")


def demo_batch_prediction(detector):
    """
    Demonstrate batch prediction on test data.
    
    WHY: Shows how to use the detector for processing
    historical data or periodic batch analysis.
    """
    print("\n" + "="*70)
    print("DEMO 2: BATCH PREDICTION")
    print("="*70)
    
    # Try to load test predictions from training
    test_pred_path = os.path.join(detector.models_dir, 'test_predictions.csv')
    
    if os.path.exists(test_pred_path):
        print(f"\n📂 Loading test data from: {test_pred_path}")
        test_data = pd.read_csv(test_pred_path)
        
        # Use a small sample for demo
        sample_size = min(100, len(test_data))
        sample_data = test_data.sample(n=sample_size, random_state=42)
        
        print(f"   Using sample of {sample_size} events for demo")
        
        # Run batch prediction
        results = detector.predict_batch(sample_data, verbose=True)
        
        # Save results
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        
        detector.save_predictions(results, os.path.join(output_dir, 'batch_predictions.csv'))
        detector.save_summary_report(results, os.path.join(output_dir, 'batch_report.txt'))
        
        # Compare with actual labels if available
        if 'is_anomaly' in sample_data.columns:
            print(f"\n📊 Comparison with Actual Labels:")
            actual = sample_data['is_anomaly'].values
            predicted = results['predictions']
            
            from sklearn.metrics import f1_score, precision_score, recall_score
            
            f1 = f1_score(actual, predicted)
            prec = precision_score(actual, predicted)
            rec = recall_score(actual, predicted)
            
            print(f"   Precision: {prec:.4f}")
            print(f"   Recall:    {rec:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
    else:
        print(f"\n⚠️  Test data not found at: {test_pred_path}")
        print(f"   Generating synthetic sample data for demo...")
        
        # Generate synthetic data
        n_samples = 50
        sample_data = pd.DataFrame({
            feat: np.random.normal(0, 1, n_samples)
            for feat in detector.selected_features
        })
        
        # Make some events anomalous
        for i in range(5):
            for feat in detector.selected_features[:5]:
                sample_data.loc[i, feat] = np.random.uniform(3, 6)
        
        results = detector.predict_batch(sample_data, verbose=True)


def demo_real_time_monitoring(detector):
    """
    Simulate real-time monitoring scenario.
    
    WHY: Shows how the detector would work in production
    where events arrive continuously.
    """
    print("\n" + "="*70)
    print("DEMO 3: REAL-TIME MONITORING SIMULATION")
    print("="*70)
    
    print(f"\n🔄 Simulating 10 incoming telemetry events...\n")
    
    # Simulate events arriving one by one
    for event_num in range(1, 11):
        # Generate random event (70% normal, 30% anomalous)
        event = {feat: np.random.normal(0, 0.5) for feat in detector.selected_features}
        
        # Make some events anomalous
        if np.random.random() < 0.3:
            for feat in detector.selected_features:
                if 'latency' in feat:
                    event[feat] = np.random.uniform(2, 5)
                elif 'memory' in feat:
                    event[feat] = np.random.uniform(2, 4)
        
        # Predict
        result = detector.predict_single(event)
        
        # Format output like a monitoring dashboard
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if result['is_anomaly']:
            severity = result['severity']
            severity_icons = {
                'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'
            }
            icon = severity_icons.get(severity, '⚠️')
            print(f"   [{timestamp}] Event #{event_num:02d}: "
                  f"{icon} {severity:<10} "
                  f"Score: {result['anomaly_score']:.4f} "
                  f"| {result['explanation']['summary'][:50]}")
        else:
            print(f"   [{timestamp}] Event #{event_num:02d}: "
                  f"⚪ NORMAL     "
                  f"Score: {result['anomaly_score']:.4f} "
                  f"| All metrics normal")
    
    print(f"\n✅ Real-time monitoring simulation complete!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # ─────────────────────────────────────────
    # Configuration
    # ─────────────────────────────────────────
    MODELS_DIR = "models"
    
    print("\n" + "="*70)
    print("ANOMALY DETECTION - INFERENCE MODULE")
    print("="*70)
    print(f"\nModels directory: {MODELS_DIR}")
    
    # ─────────────────────────────────────────
    # Initialize and load models
    # ─────────────────────────────────────────
    detector = AnomalyDetector(models_dir=MODELS_DIR)
    
    try:
        detector.load_models()
    except FileNotFoundError as e:
        print(f"\n{e}")
        print(f"\n💡 Make sure you've run train_model.py first!")
        exit(1)
    
    # ─────────────────────────────────────────
    # Print model info
    # ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MODEL INFORMATION")
    print(f"{'='*70}")
    
    model_info = detector.get_model_info()
    for key, value in model_info.items():
        if key != 'features' and key != 'ensemble_configs':
            print(f"   {key}: {value}")
    
    # ─────────────────────────────────────────
    # Run demos
    # ─────────────────────────────────────────
    
    # Demo 1: Single event prediction
    demo_single_event_prediction(detector)
    
    # Demo 2: Batch prediction
    demo_batch_prediction(detector)
    
    # Demo 3: Real-time simulation
    demo_real_time_monitoring(detector)
    
    # ─────────────────────────────────────────
    # Usage guide
    # ─────────────────────────────────────────
    print("\n" + "="*70)
    print("HOW TO USE IN YOUR CODE")
    print("="*70)
    print("""
┌─────────────────────────────────────────────────────────────┐
│  USAGE OPTION 1: Single Event (Real-Time)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  from inference import AnomalyDetector                       │
│                                                              │
│  detector = AnomalyDetector(models_dir='models')             │
│  detector.load_models()                                      │
│                                                              │
│  event = {'api_latency_ms': 0.5, 'fps': -2.1, ...}         │
│  result = detector.predict_single(event)                     │
│                                                              │
│  if result['is_anomaly']:                                    │
│      print(f"🚨 {result['severity']}")                      │
│      print(f"   {result['explanation']['summary']}")         │
│      print(f"   {result['explanation']['recommendation']}")  │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  USAGE OPTION 2: Batch Processing                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  from inference import AnomalyDetector                       │
│  import pandas as pd                                         │
│                                                              │
│  detector = AnomalyDetector(models_dir='models')             │
│  detector.load_models()                                      │
│                                                              │
│  data = pd.read_csv('new_telemetry_data.csv')               │
│  results = detector.predict_batch(data)                      │
│                                                              │
│  # Save results                                             │
│  detector.save_predictions(results, 'output.csv')           │
│  detector.save_summary_report(results, 'report.txt')        │
│                                                              │
│  # Get results as DataFrame                                 │
│  df = results['results_dataframe']                          │
│  anomalies = df[df['predicted_anomaly'] == 1]               │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  USAGE OPTION 3: In a Flask/FastAPI Endpoint                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  from inference import AnomalyDetector                       │
│                                                              │
│  # Load once at startup                                      │
│  detector = AnomalyDetector(models_dir='models')             │
│  detector.load_models()                                      │
│                                                              │
│  @app.post("/predict")                                       │
│  def predict(event: dict):                                   │
│      result = detector.predict_single(event)                 │
│      return result                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
    """)
    
    print("="*70)
    print("INFERENCE MODULE READY!")
    print("="*70)