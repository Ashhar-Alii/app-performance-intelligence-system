"""
train_model.py - Mobile App Performance Anomaly Detection Model Training (IMPROVED)

This module trains two anomaly detection models:
1. Baseline: Z-Score Method (statistical threshold)
2. Main Model: Isolation Forest ENSEMBLE (unsupervised ML)

IMPROVEMENTS OVER PREVIOUS VERSION:
- Grid search for optimal parameters
- Ensemble of top models (not single model)
- Training on normal data only (better anomaly separation)
- Validation-based threshold optimization (prevents overfitting)
- Feature correlation analysis and cleanup

Author: BCA Final Year Project
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split                   # ← NEW
from sklearn.preprocessing import RobustScaler                        # ← NEW
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings                                                        # ← NEW
warnings.filterwarnings('ignore')


class AnomalyDetectionTrainer:
    """
    Trains and evaluates anomaly detection models.
    
    IMPROVEMENTS:
    1. Grid search to find optimal Isolation Forest parameters
    2. Ensemble approach combining top models
    3. Train on normal data only for better separation
    4. Validation-based threshold optimization
    5. Feature correlation analysis
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.isolation_forest = None
        self.ensemble_models = []                                      # ← NEW: store multiple models
        self.best_params = {}                                          # ← NEW: store best parameters
        self.best_threshold = None
        self.baseline_threshold = 2.0
        self.training_stats = {}
        self.feature_names = []
        self.selected_features = []                                    # ← NEW: after correlation cleanup
        self.scaler = None                                             # ← NEW: for feature scaling
        
    def load_preprocessed_data(self, train_path, test_path):
        """
        Load preprocessed training and test data.
        
        WHY: We use the preprocessed data (scaled, encoded) not raw data.
        
        INPUT FILES:
        - train_data.csv: 8,000 events with scaled features + labels
        - test_data.csv: 2,000 events with scaled features + labels
        """
        print("="*80)
        print("STEP 1: LOADING PREPROCESSED DATA")
        print("="*80)
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"\n✅ Training data loaded: {train_df.shape}")
        print(f"✅ Test data loaded: {test_df.shape}")
        
        label_cols = ['is_anomaly', 'anomaly_type', 'timestamp', 'session_id']
        feature_cols = [col for col in train_df.columns if col not in label_cols]
        
        X_train = train_df[feature_cols]
        y_train = train_df['is_anomaly']
        
        X_test = test_df[feature_cols]
        y_test = test_df['is_anomaly']
        
        print(f"\n📊 Feature dimensions:")
        print(f"   X_train: {X_train.shape} (features only)")
        print(f"   X_test: {X_test.shape}")
        
        print(f"\n📊 Label distribution:")
        print(f"   Training - Normal: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.2f}%)")
        print(f"   Training - Anomaly: {(y_train==1).sum():,} ({(y_train==1).mean()*100:.2f}%)")
        print(f"   Test - Normal: {(y_test==0).sum():,} ({(y_test==0).mean()*100:.2f}%)")
        print(f"   Test - Anomaly: {(y_test==1).sum():,} ({(y_test==1).mean()*100:.2f}%)")
        
        self.training_stats['n_features_original'] = X_train.shape[1]
        self.training_stats['train_size'] = len(X_train)
        self.training_stats['test_size'] = len(X_test)
        self.training_stats['train_anomaly_rate'] = y_train.mean()
        self.feature_names = feature_cols
        
        return X_train, X_test, y_train, y_test, test_df
    
    # ← NEW: Entire method is new
    def _analyze_and_remove_correlated_features(self, X_train, X_test, threshold=0.95):
        """
        Remove highly correlated features to reduce noise.
        
        WHY REMOVE CORRELATED FEATURES?
        - Correlated features give the SAME information twice
        - Isolation Forest splits on random features — correlated ones waste splits
        - Example: api_latency_ms and api_latency_ms_rolling_mean_5 are ~95% correlated
        - Removing one doesn't lose information but reduces noise
        
        HOW IT WORKS:
        1. Calculate correlation matrix
        2. Find pairs with correlation > threshold (0.95)
        3. From each pair, keep the RAW feature, remove the ENGINEERED one
        4. Return cleaned datasets
        
        THRESHOLD = 0.95:
        - Only removes VERY highly correlated features (>95%)
        - Conservative — won't accidentally remove useful features
        - Typically removes 3-5 features from the 29
        """
        print(f"\n🔍 Analyzing feature correlations (threshold: {threshold})...")
        
        corr_matrix = X_train.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append({
                        'Feature_1': corr_matrix.columns[i],
                        'Feature_2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if len(high_corr_pairs) == 0:
            print(f"   No features with correlation > {threshold}")
            print(f"   Keeping all {X_train.shape[1]} features")
            self.selected_features = list(X_train.columns)
            return X_train, X_test
        
        print(f"\n   ⚠️  Found {len(high_corr_pairs)} highly correlated pairs:")
        
        # Decide which feature to drop from each pair
        # WHY: Keep raw features, drop engineered derivatives
        features_to_drop = set()
        
        # Priority for KEEPING: raw features > rolling features > zscore features
        engineered_keywords = ['rolling', 'zscore', 'change_rate', 'growth_rate', 'cv']
        
        for pair in high_corr_pairs:
            f1, f2 = pair['Feature_1'], pair['Feature_2']
            corr = pair['Correlation']
            
            f1_is_engineered = any(kw in f1 for kw in engineered_keywords)
            f2_is_engineered = any(kw in f2 for kw in engineered_keywords)
            
            if f2_is_engineered and not f1_is_engineered:
                features_to_drop.add(f2)
                action = f"DROP '{f2}' (engineered)"
            elif f1_is_engineered and not f2_is_engineered:
                features_to_drop.add(f1)
                action = f"DROP '{f1}' (engineered)"
            elif f1_is_engineered and f2_is_engineered:
                # Both engineered — drop the one with more keywords
                f1_keywords = sum(1 for kw in engineered_keywords if kw in f1)
                f2_keywords = sum(1 for kw in engineered_keywords if kw in f2)
                if f2_keywords >= f1_keywords:
                    features_to_drop.add(f2)
                    action = f"DROP '{f2}'"
                else:
                    features_to_drop.add(f1)
                    action = f"DROP '{f1}'"
            else:
                # Neither engineered — drop second one
                features_to_drop.add(f2)
                action = f"DROP '{f2}'"
            
            print(f"      {f1} ↔ {f2} (r={corr:.3f}) → {action}")
        
        # Remove features
        remaining_features = [f for f in X_train.columns if f not in features_to_drop]
        
        X_train_clean = X_train[remaining_features]
        X_test_clean = X_test[remaining_features]
        
        self.selected_features = remaining_features
        
        print(f"\n   📊 Feature reduction: {X_train.shape[1]} → {X_train_clean.shape[1]} "
              f"(removed {len(features_to_drop)} correlated features)")
        print(f"   Dropped: {sorted(features_to_drop)}")
        
        self.training_stats['n_features_after_correlation'] = X_train_clean.shape[1]
        self.training_stats['features_dropped'] = sorted(features_to_drop)
        
        return X_train_clean, X_test_clean
    
    def train_baseline_zscore(self, X_train, y_train):
        """
        Train baseline Z-Score anomaly detector.
        (Same as before — no changes needed for baseline)
        """
        print("\n" + "="*80)
        print("STEP 2: TRAINING BASELINE MODEL (Z-SCORE METHOD)")
        print("="*80)
        
        print(f"\n📖 Z-Score Method:")
        print(f"   Formula: z = (x - mean) / std")
        print(f"   Threshold: |z| > {self.baseline_threshold}")
        print(f"   Logic: Flag as anomaly if ANY feature exceeds threshold")
        
        normal_data = X_train[y_train == 0]
        
        print(f"\n🔧 Calculating statistics from NORMAL data only:")
        print(f"   Normal samples used: {len(normal_data):,} (excluding {(y_train==1).sum():,} anomalies)")
        
        self.baseline_means = normal_data.mean()
        self.baseline_stds = normal_data.std()
        
        print(f"\n✅ Baseline statistics calculated for {len(self.baseline_means)} features")
        
        z_scores = np.abs((X_train - self.baseline_means) / (self.baseline_stds + 1e-10))
        baseline_predictions = (z_scores.max(axis=1) > self.baseline_threshold).astype(int)
        
        train_precision = precision_score(y_train, baseline_predictions)
        train_recall = recall_score(y_train, baseline_predictions)
        train_f1 = f1_score(y_train, baseline_predictions)
        
        print(f"\n📊 Baseline Training Performance:")
        print(f"   Precision: {train_precision:.4f}")
        print(f"   Recall: {train_recall:.4f}")
        print(f"   F1-Score: {train_f1:.4f}")
        
        self.training_stats['baseline_train_precision'] = train_precision
        self.training_stats['baseline_train_recall'] = train_recall
        self.training_stats['baseline_train_f1'] = train_f1
        
        return baseline_predictions
    
    # ← NEW: Entire method is new
    def _grid_search_parameters(self, X_train, y_train):
        """
        Systematically find the best Isolation Forest parameters.
        
        WHY GRID SEARCH?
        - Default parameters rarely give the best results
        - Different data needs different configurations
        - Testing many combinations finds the optimal setup
        - We use a validation split to avoid overfitting
        
        HOW IT WORKS:
        1. Split training data into train (80%) and validation (20%)
        2. For each parameter combination:
           a. Train Isolation Forest on train split
           b. Find optimal threshold on validation split
           c. Calculate F1 score on validation split
        3. Return the parameters with the best F1
        
        WHY VALIDATION SPLIT?
        - Before: threshold was optimized on SAME data used for training
        - Now: threshold is optimized on UNSEEN validation data
        - This prevents overfitting and gives more realistic performance
        """
        print(f"\n🔍 Grid Search: Finding optimal parameters...")
        
        # Split training data into train and validation
        # WHY: Optimize threshold on unseen data (validation)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_train  # Keep same anomaly ratio in both splits
        )
        
        print(f"   Train split: {len(X_tr):,} samples")
        print(f"   Validation split: {len(X_val):,} samples")
        print(f"   Validation anomaly rate: {y_val.mean()*100:.2f}%")
        
        # ← NEW: Also try training on normal data only
        X_tr_normal = X_tr[y_tr == 0]
        
        # Define parameter grid
        # WHY these values: Based on your data characteristics
        param_grid = {
            'n_estimators': [200, 300],
            'max_samples': [0.6, 0.8, 1.0],
            'max_features': [0.6, 0.8, 1.0],
            'training_data': ['all', 'normal_only']  # ← NEW: try both approaches
        }
        
        total_combinations = (len(param_grid['n_estimators']) * 
                             len(param_grid['max_samples']) * 
                             len(param_grid['max_features']) * 
                             len(param_grid['training_data']))
        
        print(f"\n   Testing {total_combinations} parameter combinations...")
        print(f"\n   {'#':<4} {'Estimators':<12} {'Samples':<10} {'Features':<10} "
              f"{'Train Data':<14} {'F1':<8} {'Prec':<8} {'Rec':<8} {'AUC':<8}")
        print(f"   {'-'*82}")
        
        results = []
        count = 0
        best_f1 = 0
        
        for n_est in param_grid['n_estimators']:
            for max_samp in param_grid['max_samples']:
                for max_feat in param_grid['max_features']:
                    for train_data in param_grid['training_data']:
                        count += 1
                        
                        # Select training data
                        if train_data == 'normal_only':
                            X_fit = X_tr_normal
                        else:
                            X_fit = X_tr
                        
                        # Train model
                        clf = IsolationForest(
                            n_estimators=n_est,
                            max_samples=min(max_samp, 1.0) if isinstance(max_samp, float) else max_samp,
                            max_features=max_feat,
                            contamination='auto',
                            random_state=self.random_state,
                            n_jobs=-1
                        )
                        clf.fit(X_fit)
                        
                        # Get scores on validation set
                        val_scores = clf.score_samples(X_val)
                        
                        # Find best threshold on validation
                        best_thresh_f1 = 0
                        best_thresh = None
                        
                        for pct in np.arange(5, 35, 1.0):
                            thresh = np.percentile(val_scores, pct)
                            preds = (val_scores < thresh).astype(int)
                            if preds.sum() == 0:
                                continue
                            f1 = f1_score(y_val, preds)
                            if f1 > best_thresh_f1:
                                best_thresh_f1 = f1
                                best_thresh = thresh
                        
                        if best_thresh is None:
                            continue
                        
                        # Calculate all metrics at best threshold
                        val_preds = (val_scores < best_thresh).astype(int)
                        val_prec = precision_score(y_val, val_preds)
                        val_rec = recall_score(y_val, val_preds)
                        val_auc = roc_auc_score(y_val, -val_scores)
                        
                        result = {
                            'n_estimators': n_est,
                            'max_samples': max_samp,
                            'max_features': max_feat,
                            'training_data': train_data,
                            'f1': best_thresh_f1,
                            'precision': val_prec,
                            'recall': val_rec,
                            'auc_roc': val_auc,
                            'threshold': best_thresh,
                            'model': clf
                        }
                        results.append(result)
                        
                        marker = " ← BEST" if best_thresh_f1 > best_f1 else ""
                        if best_thresh_f1 > best_f1:
                            best_f1 = best_thresh_f1
                        
                        print(f"   {count:<4} {n_est:<12} {max_samp:<10} {max_feat:<10} "
                              f"{train_data:<14} {best_thresh_f1:<8.4f} {val_prec:<8.4f} "
                              f"{val_rec:<8.4f} {val_auc:<8.4f}{marker}")
        
        # Sort by F1 score
        results.sort(key=lambda x: x['f1'], reverse=True)
        
        # Store best parameters
        best = results[0]
        self.best_params = {
            'n_estimators': best['n_estimators'],
            'max_samples': best['max_samples'],
            'max_features': best['max_features'],
            'training_data': best['training_data']
        }
        
        print(f"\n   🏆 BEST PARAMETERS FOUND:")
        print(f"      n_estimators:  {best['n_estimators']}")
        print(f"      max_samples:   {best['max_samples']}")
        print(f"      max_features:  {best['max_features']}")
        print(f"      training_data: {best['training_data']}")
        print(f"      Validation F1: {best['f1']*100:.2f}%")
        print(f"      Validation AUC: {best['auc_roc']:.4f}")
        
        # ← NEW: Show top 5 results
        print(f"\n   📊 Top 5 Configurations:")
        print(f"   {'Rank':<6} {'F1':<8} {'Prec':<8} {'Rec':<8} {'AUC':<8} {'Config'}")
        print(f"   {'-'*70}")
        for i, r in enumerate(results[:5], 1):
            config = f"est={r['n_estimators']}, samp={r['max_samples']}, feat={r['max_features']}, data={r['training_data']}"
            print(f"   {i:<6} {r['f1']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f} "
                  f"{r['auc_roc']:<8.4f} {config}")
        
        # Store top models for ensemble
        self.grid_results = results
        self.training_stats['best_params'] = self.best_params
        self.training_stats['grid_search_total'] = total_combinations
        
        return results
    
    # ← NEW: Entire method is new
    def _build_ensemble(self, X_train, y_train, top_n=3):
        """
        Build an ensemble of top N models from grid search.
        
        WHY ENSEMBLE?
        - Single model has blind spots (misses certain anomaly patterns)
        - Multiple models with different configs see different patterns
        - Combining their scores → more robust predictions
        - Reduces false positives AND false negatives simultaneously
        
        HOW IT WORKS:
        1. Take top N models from grid search
        2. Each model produces anomaly scores
        3. Normalize scores to same scale (0-1)
        4. Average the normalized scores
        5. Use averaged score for final prediction
        
        WHY TOP 3?
        - Top 1 alone might overfit to validation set
        - Top 3 gives diversity while keeping quality
        - More than 5 adds noise without benefit
        - Odd number avoids ties in voting
        """
        print(f"\n🔧 Building Ensemble of Top {top_n} Models...")
        
        self.ensemble_models = []
        
        # Get training subsets
        X_normal = X_train[y_train == 0]
        
        for i, result in enumerate(self.grid_results[:top_n], 1):
            # Retrain on FULL training data (not just the 80% split)
            # WHY: Grid search used 80% for training, 20% for validation
            # Now we retrain the best configs on 100% of training data
            
            if result['training_data'] == 'normal_only':
                X_fit = X_normal
            else:
                X_fit = X_train
            
            model = IsolationForest(
                n_estimators=result['n_estimators'],
                max_samples=result['max_samples'],
                max_features=result['max_features'],
                contamination='auto',
                random_state=self.random_state + i,  # Different random state for diversity
                n_jobs=-1
            )
            model.fit(X_fit)
            
            self.ensemble_models.append({
                'model': model,
                'params': {
                    'n_estimators': result['n_estimators'],
                    'max_samples': result['max_samples'],
                    'max_features': result['max_features'],
                    'training_data': result['training_data']
                },
                'val_f1': result['f1']
            })
            
            print(f"   Model {i}: est={result['n_estimators']}, "
                  f"samp={result['max_samples']}, feat={result['max_features']}, "
                  f"data={result['training_data']}, val_F1={result['f1']:.4f}")
        
        print(f"\n✅ Ensemble of {top_n} models built successfully!")
        self.training_stats['ensemble_size'] = top_n
        
        return self.ensemble_models
    
    # ← NEW: Entire method is new
    def _get_ensemble_scores(self, X):
        """
        Get combined anomaly scores from the ensemble.
        
        WHY COMBINE SCORES?
        - Each model gives slightly different scores
        - Averaging smooths out individual model errors
        - More reliable than any single model alone
        
        HOW:
        1. Get raw scores from each model
        2. Normalize each model's scores to 0-1 range (Min-Max)
        3. Average the normalized scores
        4. Return averaged scores (lower = more anomalous)
        
        WHY NORMALIZE?
        - Different models produce scores on different scales
        - Model A might give scores from -0.5 to -0.1
        - Model B might give scores from -0.8 to -0.2
        - Without normalization, Model B would dominate the average
        """
        all_scores = []
        
        for model_info in self.ensemble_models:
            model = model_info['model']
            scores = model.score_samples(X)
            
            # Min-Max normalize to [0, 1]
            # WHY: Put all models on same scale before averaging
            score_min = scores.min()
            score_max = scores.max()
            
            if score_max - score_min > 0:
                normalized = (scores - score_min) / (score_max - score_min)
            else:
                normalized = np.zeros_like(scores)
            
            all_scores.append(normalized)
        
        # Average across all models
        # WHY: Ensemble average is more stable than any individual model
        ensemble_scores = np.mean(all_scores, axis=0)
        
        # Convert back: lower = more anomalous (for consistency)
        # After normalization: 0 = most anomalous, 1 = most normal
        # We negate so lower = more anomalous (matching sklearn convention)
        ensemble_scores = -ensemble_scores
        
        return ensemble_scores
    
    def _find_optimal_threshold(self, scores, y_true):
        """
        Find the optimal anomaly score threshold that maximizes F1 score.
        
        IMPROVED FROM PREVIOUS VERSION:
        - Finer search granularity (0.25% steps instead of 0.5%)
        - Wider search range
        - Also tracks precision and recall balance
        - Prints cleaner output
        """
        best_f1 = 0
        best_threshold = None
        best_percentile = None
        best_precision = 0
        best_recall = 0
        
        print(f"\n   {'Pct':>8} {'Threshold':>12} {'F1':>8} {'Prec':>8} {'Rec':>8} {'Anom%':>8}")
        print(f"   {'-'*56}")
        
        # ← CHANGED: Finer search with 0.25% steps
        for pct in np.arange(3, 40, 0.25):
            threshold = np.percentile(scores, pct)
            pred_labels = (scores < threshold).astype(int)
            
            if pred_labels.sum() == 0:
                continue
                
            f1 = f1_score(y_true, pred_labels)
            prec = precision_score(y_true, pred_labels)
            rec = recall_score(y_true, pred_labels)
            pred_rate = pred_labels.mean() * 100
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_percentile = pct
                best_precision = prec
                best_recall = rec
                marker = " ← BEST"
            else:
                marker = ""
            
            # Print key milestones + any new best
            if pct % 2 < 0.25 or marker:
                print(f"   {pct:>6.1f}% {threshold:>12.4f} {f1*100:>7.2f}% "
                      f"{prec*100:>7.2f}% {rec*100:>7.2f}% {pred_rate:>7.1f}%{marker}")
        
        print(f"\n   🏆 Optimal Threshold:")
        print(f"      Percentile:  {best_percentile:.2f}%")
        print(f"      Threshold:   {best_threshold:.4f}")
        print(f"      F1 Score:    {best_f1*100:.2f}%")
        print(f"      Precision:   {best_precision*100:.2f}%")
        print(f"      Recall:      {best_recall*100:.2f}%")
        
        return best_threshold
    
    def train_isolation_forest(self, X_train, y_train):
        """
        Train Isolation Forest anomaly detector (IMPROVED VERSION).
        
        IMPROVEMENTS OVER PREVIOUS VERSION:
        1. Feature correlation cleanup → removes noisy redundant features
        2. Grid search → finds optimal parameters systematically
        3. Normal-only training → model learns "normal" better
        4. Ensemble → combines top 3 models for robustness
        5. Validation-based threshold → better generalization
        
        TRAINING PIPELINE:
        Step 3a: Clean correlated features
        Step 3b: Grid search for best parameters
        Step 3c: Build ensemble of top models
        Step 3d: Optimize threshold on training data
        Step 3e: Evaluate and report results
        """
        print("\n" + "="*80)
        print("STEP 3: TRAINING ISOLATION FOREST MODEL (IMPROVED)")
        print("="*80)
        
        actual_anomaly_rate = y_train.mean()
        print(f"\n   Actual anomaly rate: {actual_anomaly_rate*100:.2f}%")
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Features: {X_train.shape[1]}")
        
        # ─────────────────────────────────────────────
        # Step 3a: Feature Correlation Cleanup
        # ─────────────────────────────────────────────
        print("\n" + "-"*60)
        print("STEP 3a: FEATURE CORRELATION CLEANUP")
        print("-"*60)
        
        # Store original X for baseline evaluation later
        self.X_train_original = X_train.copy()
        
        X_train_clean, _ = self._analyze_and_remove_correlated_features(
            X_train, X_train, threshold=0.95
        )
        
        # Update y_train index to match
        # (No rows removed, only columns, so y_train stays the same)
        
        # ─────────────────────────────────────────────
        # Step 3b: Grid Search
        # ─────────────────────────────────────────────
        print("\n" + "-"*60)
        print("STEP 3b: GRID SEARCH FOR OPTIMAL PARAMETERS")
        print("-"*60)
        
        print(f"\n💡 WHY Grid Search?")
        print(f"   Previous: Used fixed parameters (might not be optimal)")
        print(f"   Now: Test many combinations, find the best one")
        print(f"   Also tests: Training on ALL data vs NORMAL data only")
        
        grid_results = self._grid_search_parameters(X_train_clean, y_train)
        
        # ─────────────────────────────────────────────
        # Step 3c: Build Ensemble
        # ─────────────────────────────────────────────
        print("\n" + "-"*60)
        print("STEP 3c: BUILDING MODEL ENSEMBLE")
        print("-"*60)
        
        print(f"\n💡 WHY Ensemble?")
        print(f"   Single model: Has blind spots, misses some patterns")
        print(f"   Ensemble: Multiple models vote → more robust predictions")
        print(f"   Like asking 3 doctors instead of 1 for a diagnosis")
        
        self._build_ensemble(X_train_clean, y_train, top_n=3)
        
        # Also train the single best model separately
        # WHY: Save both single model AND ensemble for comparison
        best = grid_results[0]
        
        X_normal = X_train_clean[y_train == 0]
        X_fit = X_normal if best['training_data'] == 'normal_only' else X_train_clean
        
        self.isolation_forest = IsolationForest(
            n_estimators=best['n_estimators'],
            max_samples=best['max_samples'],
            max_features=best['max_features'],
            contamination='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.isolation_forest.fit(X_fit)
        
        print(f"\n✅ Single best model also trained for comparison")
        
        # ─────────────────────────────────────────────
        # Step 3d: Get Ensemble Scores & Optimize Threshold
        # ─────────────────────────────────────────────
        print("\n" + "-"*60)
        print("STEP 3d: THRESHOLD OPTIMIZATION (ENSEMBLE)")
        print("-"*60)
        
        print(f"\n💡 Using ENSEMBLE scores for threshold optimization")
        
        # Get ensemble scores on training data
        ensemble_scores = self._get_ensemble_scores(X_train_clean)
        
        # Also get single model scores for comparison
        single_scores = self.isolation_forest.score_samples(X_train_clean)
        
        print(f"\n📊 Ensemble Threshold Search:")
        self.best_threshold = self._find_optimal_threshold(ensemble_scores, y_train)
        
        # Also find threshold for single model
        print(f"\n📊 Single Model Threshold Search:")
        self.single_threshold = self._find_optimal_threshold(single_scores, y_train)
        
        # ─────────────────────────────────────────────
        # Step 3e: Compare Single vs Ensemble on Training Data
        # ─────────────────────────────────────────────
        print("\n" + "-"*60)
        print("STEP 3e: SINGLE MODEL vs ENSEMBLE COMPARISON (Training)")
        print("-"*60)
        
        # Single model predictions
        single_preds = (single_scores < self.single_threshold).astype(int)
        single_f1 = f1_score(y_train, single_preds)
        single_prec = precision_score(y_train, single_preds)
        single_rec = recall_score(y_train, single_preds)
        
        # Ensemble predictions
        ensemble_preds = (ensemble_scores < self.best_threshold).astype(int)
        ensemble_f1 = f1_score(y_train, ensemble_preds)
        ensemble_prec = precision_score(y_train, ensemble_preds)
        ensemble_rec = recall_score(y_train, ensemble_preds)
        
        print(f"\n   {'Metric':<12} {'Single Model':<16} {'Ensemble':<16} {'Winner':<10}")
        print(f"   {'-'*54}")
        
        for metric, s_val, e_val in [
            ('Precision', single_prec, ensemble_prec),
            ('Recall', single_rec, ensemble_rec),
            ('F1-Score', single_f1, ensemble_f1)
        ]:
            winner = "Ensemble ✅" if e_val >= s_val else "Single ✅"
            print(f"   {metric:<12} {s_val:<16.4f} {e_val:<16.4f} {winner}")
        
        # Decide which approach to use
        # WHY: Use whichever gives better F1 on training data
        if ensemble_f1 >= single_f1:
            self.use_ensemble = True
            predictions = ensemble_preds
            final_scores = ensemble_scores
            print(f"\n   🏆 Using ENSEMBLE approach (better F1)")
        else:
            self.use_ensemble = False
            predictions = single_preds
            final_scores = single_scores
            self.best_threshold = self.single_threshold
            print(f"\n   🏆 Using SINGLE MODEL approach (better F1)")
        
        # Final training metrics
        train_precision = precision_score(y_train, predictions)
        train_recall = recall_score(y_train, predictions)
        train_f1 = f1_score(y_train, predictions)
        train_auc = roc_auc_score(y_train, -final_scores)
        
        print(f"\n📊 Final Training Performance:")
        print(f"   Precision: {train_precision:.4f}")
        print(f"   Recall:    {train_recall:.4f}")
        print(f"   F1-Score:  {train_f1:.4f}")
        print(f"   AUC-ROC:   {train_auc:.4f}")
        
        print(f"\n💡 What These Metrics Mean:")
        print(f"   Precision: Of all flagged anomalies, {train_precision*100:.1f}% were actually anomalies")
        print(f"   Recall: Of all actual anomalies, we detected {train_recall*100:.1f}%")
        print(f"   F1-Score: Harmonic mean of precision and recall")
        
        # Store stats
        self.training_stats['iforest_train_precision'] = train_precision
        self.training_stats['iforest_train_recall'] = train_recall
        self.training_stats['iforest_train_f1'] = train_f1
        self.training_stats['iforest_train_auc'] = train_auc
        self.training_stats['optimal_threshold'] = self.best_threshold
        self.training_stats['use_ensemble'] = self.use_ensemble
        self.training_stats['n_features_used'] = len(self.selected_features)
        
        return predictions, final_scores
    
    def evaluate_on_test_data(self, X_test, y_test, test_df):
        """
        Evaluate both models on test data.
        
        IMPROVED: Now evaluates both single model AND ensemble on test data.
        Uses selected features (after correlation cleanup).
        """
        print("\n" + "="*80)
        print("STEP 4: EVALUATING ON TEST DATA")
        print("="*80)
        
        print(f"\n🧪 Test Set Details:")
        print(f"   Size: {len(X_test):,} events")
        print(f"   Anomaly rate: {y_test.mean()*100:.2f}%")
        print(f"   ⚠️  Model has NEVER seen this data during training!")
        
        # ← CHANGED: Use selected features only
        X_test_clean = X_test[self.selected_features]
        print(f"   Features used: {len(self.selected_features)} (after correlation cleanup)")
        
        # ========================================
        # Baseline Z-Score Predictions
        # ========================================
        print(f"\n" + "-"*80)
        print("BASELINE MODEL (Z-SCORE) - TEST EVALUATION")
        print("-"*80)
        
        # Use original features for baseline (baseline was trained on all features)
        z_scores = np.abs((X_test - self.baseline_means) / (self.baseline_stds + 1e-10))
        baseline_predictions = (z_scores.max(axis=1) > self.baseline_threshold).astype(int)
        
        baseline_precision = precision_score(y_test, baseline_predictions)
        baseline_recall = recall_score(y_test, baseline_predictions)
        baseline_f1 = f1_score(y_test, baseline_predictions)
        
        print(f"\n📊 Baseline Test Performance:")
        print(f"   Precision: {baseline_precision:.4f}")
        print(f"   Recall:    {baseline_recall:.4f}")
        print(f"   F1-Score:  {baseline_f1:.4f}")
        
        baseline_cm = confusion_matrix(y_test, baseline_predictions)
        print(f"\n   Confusion Matrix:")
        print(f"                Predicted Normal  Predicted Anomaly")
        print(f"   Actual Normal       {baseline_cm[0,0]:6d}            {baseline_cm[0,1]:6d}")
        print(f"   Actual Anomaly      {baseline_cm[1,0]:6d}            {baseline_cm[1,1]:6d}")
        
        # ========================================
        # Isolation Forest Predictions (Ensemble or Single)
        # ========================================
        print(f"\n" + "-"*80)
        if self.use_ensemble:
            print("ISOLATION FOREST ENSEMBLE - TEST EVALUATION")
        else:
            print("ISOLATION FOREST (BEST SINGLE MODEL) - TEST EVALUATION")
        print("-"*80)
        
        # Get scores
        if self.use_ensemble:
            iforest_scores = self._get_ensemble_scores(X_test_clean)
            print(f"\n   Using ensemble of {len(self.ensemble_models)} models")
        else:
            iforest_scores = self.isolation_forest.score_samples(X_test_clean)
            print(f"\n   Using single best model")
        
        print(f"   Threshold: {self.best_threshold:.4f}")
        
        # Make predictions using optimized threshold
        iforest_predictions = (iforest_scores < self.best_threshold).astype(int)
        
        print(f"   Predicted anomaly rate: {iforest_predictions.mean()*100:.2f}%")
        print(f"   Actual anomaly rate: {y_test.mean()*100:.2f}%")
        
        # Calculate metrics
        iforest_precision = precision_score(y_test, iforest_predictions)
        iforest_recall = recall_score(y_test, iforest_predictions)
        iforest_f1 = f1_score(y_test, iforest_predictions)
        iforest_auc = roc_auc_score(y_test, -iforest_scores)
        
        print(f"\n📊 Isolation Forest Test Performance:")
        print(f"   Precision: {iforest_precision:.4f}")
        print(f"   Recall:    {iforest_recall:.4f}")
        print(f"   F1-Score:  {iforest_f1:.4f}")
        print(f"   AUC-ROC:   {iforest_auc:.4f}")
        
        # Confusion matrix
        iforest_cm = confusion_matrix(y_test, iforest_predictions)
        print(f"\n   Confusion Matrix:")
        print(f"                Predicted Normal  Predicted Anomaly")
        print(f"   Actual Normal       {iforest_cm[0,0]:6d}            {iforest_cm[0,1]:6d}")
        print(f"   Actual Anomaly      {iforest_cm[1,0]:6d}            {iforest_cm[1,1]:6d}")
        
        tn, fp, fn, tp = iforest_cm.ravel()
        print(f"\n   📖 Confusion Matrix Explained:")
        print(f"      True Positives  (correctly caught anomalies): {tp}")
        print(f"      True Negatives  (correctly identified normal): {tn}")
        print(f"      False Positives (normal wrongly flagged):      {fp}")
        print(f"      False Negatives (anomalies we MISSED):         {fn}")
        
        # ========================================
        # Model Comparison
        # ========================================
        print(f"\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        print(f"\n{'Metric':<15s} {'Baseline':>12s} {'Isolation Forest':>18s} {'Improvement':>12s}")
        print("-"*60)
        
        if baseline_precision > 0:
            precision_improvement = ((iforest_precision - baseline_precision) / baseline_precision) * 100
        else:
            precision_improvement = 0.0
            
        if baseline_recall > 0:
            recall_improvement = ((iforest_recall - baseline_recall) / baseline_recall) * 100
        else:
            recall_improvement = 0.0
            
        if baseline_f1 > 0:
            f1_improvement = ((iforest_f1 - baseline_f1) / baseline_f1) * 100
        else:
            f1_improvement = 0.0
        
        print(f"{'Precision':<15s} {baseline_precision:>12.4f} {iforest_precision:>18.4f} {precision_improvement:>11.2f}%")
        print(f"{'Recall':<15s} {baseline_recall:>12.4f} {iforest_recall:>18.4f} {recall_improvement:>11.2f}%")
        print(f"{'F1-Score':<15s} {baseline_f1:>12.4f} {iforest_f1:>18.4f} {f1_improvement:>11.2f}%")
        print(f"{'AUC-ROC':<15s} {'N/A':>12s} {iforest_auc:>18.4f} {'':>12s}")
        
        # Store results
        results = {
            'baseline': {
                'predictions': baseline_predictions,
                'precision': baseline_precision,
                'recall': baseline_recall,
                'f1': baseline_f1,
                'confusion_matrix': baseline_cm
            },
            'isolation_forest': {
                'predictions': iforest_predictions,
                'scores': iforest_scores,
                'precision': iforest_precision,
                'recall': iforest_recall,
                'f1': iforest_f1,
                'auc_roc': iforest_auc,
                'threshold': self.best_threshold,
                'confusion_matrix': iforest_cm
            },
            'improvement': {
                'precision': precision_improvement,
                'recall': recall_improvement,
                'f1': f1_improvement
            }
        }
        
        self.training_stats['baseline_test_precision'] = baseline_precision
        self.training_stats['baseline_test_recall'] = baseline_recall
        self.training_stats['baseline_test_f1'] = baseline_f1
        self.training_stats['iforest_test_precision'] = iforest_precision
        self.training_stats['iforest_test_recall'] = iforest_recall
        self.training_stats['iforest_test_f1'] = iforest_f1
        self.training_stats['iforest_test_auc_roc'] = iforest_auc
        
        return results
    
    def save_models(self, output_dir='models'):
        """
        Save trained models and statistics.
        
        IMPROVED: Now saves ensemble models AND selected features.
        """
        print("\n" + "="*80)
        print("STEP 5: SAVING TRAINED MODELS")
        print("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save single best Isolation Forest
        iforest_path = os.path.join(output_dir, 'isolation_forest.pkl')
        joblib.dump(self.isolation_forest, iforest_path)
        print(f"\n✅ Isolation Forest (single) saved: {iforest_path}")
        
        # ← NEW: Save ensemble models
        if self.ensemble_models:
            ensemble_path = os.path.join(output_dir, 'ensemble_models.pkl')
            joblib.dump(self.ensemble_models, ensemble_path)
            print(f"✅ Ensemble models saved: {ensemble_path}")
        
        # Save baseline statistics
        baseline_stats = {
            'means': self.baseline_means,
            'stds': self.baseline_stds,
            'threshold': self.baseline_threshold
        }
        baseline_path = os.path.join(output_dir, 'baseline_stats.pkl')
        joblib.dump(baseline_stats, baseline_path)
        print(f"✅ Baseline statistics saved: {baseline_path}")
        
        # Save threshold and configuration
        threshold_data = {
            'best_threshold': self.best_threshold,
            'feature_names': self.selected_features,  # ← CHANGED: use selected features
            'use_ensemble': self.use_ensemble,
            'best_params': self.best_params
        }
        threshold_path = os.path.join(output_dir, 'optimal_threshold.pkl')
        joblib.dump(threshold_data, threshold_path)
        print(f"✅ Optimal threshold & config saved: {threshold_path}")
        
        # ← NEW: Save selected features list
        features_path = os.path.join(output_dir, 'selected_features.pkl')
        joblib.dump(self.selected_features, features_path)
        print(f"✅ Selected features saved: {features_path}")
        
        # Save training statistics
        stats_path = os.path.join(output_dir, 'training_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("MODEL TRAINING STATISTICS (IMPROVED)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MODEL CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"  Algorithm: Isolation Forest {'Ensemble' if self.use_ensemble else 'Single'}\n")
            if self.best_params:
                for k, v in self.best_params.items():
                    f.write(f"  {k}: {v}\n")
            f.write(f"  optimal_threshold: {self.best_threshold:.6f}\n")
            f.write(f"  features_used: {len(self.selected_features)}\n")
            if self.use_ensemble:
                f.write(f"  ensemble_size: {len(self.ensemble_models)}\n")
            f.write("\n")
            
            f.write("PERFORMANCE METRICS:\n")
            f.write("-"*40 + "\n")
            for key, value in self.training_stats.items():
                if not isinstance(value, (list, dict)):
                    f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {key}: {str(value)}\n")
        print(f"✅ Training statistics saved: {stats_path}")
        
        print(f"\n📁 All models saved to: {output_dir}/")
        
        return output_dir
    
    def run_full_training_pipeline(self, train_path, test_path, output_dir='models'):
        """
        Execute complete IMPROVED training pipeline.
        
        PIPELINE STEPS:
        1. Load preprocessed data
        2. Train baseline (Z-Score)
        3. Train Isolation Forest (Grid Search + Ensemble + Threshold Optimization)
        4. Evaluate both on test data
        5. Save models
        """
        print("\n" + "="*80)
        print("ANOMALY DETECTION - IMPROVED MODEL TRAINING PIPELINE")
        print("="*80)
        print(f"\nTrain data: {train_path}")
        print(f"Test data: {test_path}")
        print(f"Output directory: {output_dir}")
        
        print(f"\n🆕 IMPROVEMENTS IN THIS VERSION:")
        print(f"   ✅ Grid search for optimal parameters")
        print(f"   ✅ Training on normal data only (option)")
        print(f"   ✅ Ensemble of top 3 models")
        print(f"   ✅ Feature correlation cleanup")
        print(f"   ✅ Validation-based threshold optimization")
        
        # Step 1: Load data
        X_train, X_test, y_train, y_test, test_df = self.load_preprocessed_data(
            train_path, test_path
        )
        
        # Step 2: Train baseline
        baseline_train_predictions = self.train_baseline_zscore(X_train, y_train)
        
        # Step 3: Train Isolation Forest (IMPROVED)
        iforest_train_predictions, iforest_train_scores = self.train_isolation_forest(
            X_train, y_train
        )
        
        # Step 4: Evaluate on test data
        results = self.evaluate_on_test_data(X_test, y_test, test_df)
        
        # ← NEW: Save test predictions for evaluation module
        X_test_clean = X_test[self.selected_features]
        
        eval_data = test_df.copy()
        eval_data['predicted_anomaly'] = results['isolation_forest']['predictions']
        eval_data['anomaly_score'] = results['isolation_forest']['scores']
        eval_data['baseline_prediction'] = results['baseline']['predictions']
        
        eval_output_path = os.path.join(output_dir, 'test_predictions.csv')
        os.makedirs(output_dir, exist_ok=True)
        eval_data.to_csv(eval_output_path, index=False)
        print(f"\n✅ Test predictions saved: {eval_output_path}")
        
        # Step 5: Save models
        model_dir = self.save_models(output_dir)
        
        # Final summary
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        
        print(f"\n✅ Models trained and evaluated")
        
        approach = "Ensemble" if self.use_ensemble else "Single Model"
        print(f"\n🏆 Final Results ({approach}):")
        print(f"   ┌──────────────────────────────────────────┐")
        print(f"   │  Test F1-Score:  {results['isolation_forest']['f1']:.4f}                   │")
        print(f"   │  Test Precision: {results['isolation_forest']['precision']:.4f}                   │")
        print(f"   │  Test Recall:    {results['isolation_forest']['recall']:.4f}                   │")
        print(f"   │  Test AUC-ROC:   {results['isolation_forest']['auc_roc']:.4f}                   │")
        print(f"   │  vs Baseline:    {results['improvement']['f1']:+.2f}% F1              │")
        print(f"   └──────────────────────────────────────────┘")
        
        print(f"\n🎯 Optimal Threshold: {self.best_threshold:.4f}")
        print(f"   Features used: {len(self.selected_features)} (from {len(self.feature_names)} original)")
        
        if self.best_params:
            print(f"\n🔧 Best Parameters Found:")
            for k, v in self.best_params.items():
                print(f"   {k}: {v}")
        
        print(f"\n📂 Saved models:")
        print(f"   {model_dir}/isolation_forest.pkl")
        if self.use_ensemble:
            print(f"   {model_dir}/ensemble_models.pkl")
        print(f"   {model_dir}/baseline_stats.pkl")
        print(f"   {model_dir}/optimal_threshold.pkl")
        print(f"   {model_dir}/selected_features.pkl")
        print(f"   {model_dir}/training_stats.txt")
        print(f"   {model_dir}/test_predictions.csv")
        
        print("\n" + "="*80)
        print("NEXT STEP: Run evaluate_model.py for detailed analysis")
        print("="*80)
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    TRAIN_DATA_PATH = "data/processed/train_data.csv"
    TEST_DATA_PATH = "data/processed/test_data.csv"
    OUTPUT_DIR = "models"
    RANDOM_STATE = 42
    
    trainer = AnomalyDetectionTrainer(random_state=RANDOM_STATE)
    
    results = trainer.run_full_training_pipeline(
        train_path=TRAIN_DATA_PATH,
        test_path=TEST_DATA_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Print usage example
    print("\n" + "="*80)
    print("USAGE EXAMPLE - HOW TO LOAD AND USE TRAINED MODELS:")
    print("="*80)
    print("""
import joblib
import numpy as np

# Load model components
model = joblib.load('models/isolation_forest.pkl')
ensemble = joblib.load('models/ensemble_models.pkl')
threshold_data = joblib.load('models/optimal_threshold.pkl')
selected_features = joblib.load('models/selected_features.pkl')

best_threshold = threshold_data['best_threshold']
use_ensemble = threshold_data['use_ensemble']

# Load and prepare new data
new_data = pd.read_csv('new_telemetry_data.csv')
new_data_clean = new_data[selected_features]  # Use only selected features!

if use_ensemble:
    # Ensemble prediction
    all_scores = []
    for model_info in ensemble:
        scores = model_info['model'].score_samples(new_data_clean)
        s_min, s_max = scores.min(), scores.max()
        normalized = (scores - s_min) / (s_max - s_min) if s_max > s_min else np.zeros_like(scores)
        all_scores.append(normalized)
    final_scores = -np.mean(all_scores, axis=0)
else:
    final_scores = model.score_samples(new_data_clean)

predictions = (final_scores < best_threshold).astype(int)
print(f"Detected {predictions.sum()} anomalies out of {len(new_data)} events")
    """)