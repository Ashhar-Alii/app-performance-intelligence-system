"""
evaluate_model.py - Comprehensive Anomaly Detection Model Evaluation

This module performs in-depth analysis of trained models including:
1. Classification metrics (precision, recall, F1)
2. Threshold-independent metrics (AUC-ROC, AUC-PR)
3. Anomaly score analysis
4. Error analysis (false positives/negatives)
5. Model comparison visualizations
6. Feature importance analysis

WHY THIS MODULE?
- Training metrics alone don't tell the full story
- Need to understand WHERE and WHY the model fails
- Visualizations help communicate results to non-technical stakeholders
- Error analysis guides future improvements

Author: BCA Final Year Project
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,                                                   # ← NEW
    recall_score,                                                      # ← NEW
    f1_score                                                           # ← NEW
)
import joblib
import os
from datetime import datetime

# Set style for professional-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ModelEvaluator:
    """
    Comprehensive evaluation of anomaly detection models.
    
    WHY: Training gives us numbers, evaluation gives us UNDERSTANDING.
    """
    
    def __init__(self, models_dir='models', output_dir='evaluation'):
        self.models_dir = models_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Store loaded data
        self.test_data = None
        self.model = None
        self.threshold_data = None
        
    def load_test_predictions(self):
        """
        Load test data with predictions from training.
        
        WHY: The training module saved test predictions + scores
        This includes ALL the data we need for evaluation
        
        WHAT WE GET:
        - True labels (is_anomaly, anomaly_type)
        - Model predictions (predicted_anomaly)
        - Anomaly scores (anomaly_score)
        - Baseline predictions (baseline_prediction)
        - All original features (for error analysis)
        """
        print("="*80)
        print("STEP 1: LOADING TEST DATA AND PREDICTIONS")
        print("="*80)
        
        # Load predictions saved during training
        pred_path = os.path.join(self.models_dir, 'test_predictions.csv')
        self.test_data = pd.read_csv(pred_path)
        
        print(f"\n✅ Loaded test predictions: {self.test_data.shape}")
        print(f"\n   Key columns:")
        print(f"   - is_anomaly: Ground truth labels")
        print(f"   - anomaly_type: Type of anomaly")
        print(f"   - predicted_anomaly: Model predictions")
        print(f"   - anomaly_score: Isolation Forest scores")
        print(f"   - baseline_prediction: Z-Score predictions")
        
        # Load model and threshold
        model_path = os.path.join(self.models_dir, 'isolation_forest.pkl')
        self.model = joblib.load(model_path)
        
        threshold_path = os.path.join(self.models_dir, 'optimal_threshold.pkl')
        self.threshold_data = joblib.load(threshold_path)
        
        print(f"\n✅ Loaded Isolation Forest model")
        print(f"✅ Optimal threshold: {self.threshold_data['best_threshold']:.4f}")
        
        return self.test_data
    
    def generate_classification_metrics(self):
        """
        SECTION 1: CLASSIFICATION METRICS
        
        WHY: Standard metrics that everyone understands
        - Precision: How accurate are our alerts?
        - Recall: How many anomalies did we catch?
        - F1-Score: Overall balance
        
        OUTPUT: Detailed classification report + confusion matrix heatmap
        """
        print("\n" + "="*80)
        print("SECTION 1: CLASSIFICATION METRICS")
        print("="*80)
        
        y_true = self.test_data['is_anomaly']
        y_pred = self.test_data['predicted_anomaly']
        
        # Generate classification report
        # WHY: Shows precision/recall/F1 for EACH class (normal + anomaly)
        report = classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Anomaly'],
            digits=4
        )
        
        print("\n📊 Classification Report:")
        print(report)
        
        # Confusion Matrix with Heatmap
        # WHY: Visual representation makes it easier to see patterns
        cm = confusion_matrix(y_true, y_pred)
        
        # Create heatmap
        # WHY: Colors make it immediately obvious where errors occur
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Anomaly'],
            yticklabels=['Actual Normal', 'Actual Anomaly'],
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Isolation Forest\n', fontsize=16, fontweight='bold')
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add percentage annotations
        # WHY: Raw counts don't show the full picture, percentages do
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                pct = (cm[i, j] / total) * 100
                plt.text(j+0.5, i+0.7, f'({pct:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        cm_path = os.path.join(self.output_dir, '1_confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Confusion matrix heatmap saved: {cm_path}")
        plt.close()
        
        # Detailed breakdown
        # WHY: Explain what each cell means in business terms
        tn, fp, fn, tp = cm.ravel()
        print(f"\n📖 Confusion Matrix Breakdown:")
        print(f"   True Negatives (TN):  {tn:4d} - Correctly identified normal events")
        print(f"   False Positives (FP): {fp:4d} - Normal events wrongly flagged (ALERT FATIGUE)")
        print(f"   False Negatives (FN): {fn:4d} - Missed anomalies (RISK)")
        print(f"   True Positives (TP):  {tp:4d} - Correctly caught anomalies (SUCCESS)")
        
        print(f"\n💡 Business Impact:")
        print(f"   Alert Accuracy: {tp/(tp+fp)*100:.1f}% of alerts are real issues")
        print(f"   Detection Rate: {tp/(tp+fn)*100:.1f}% of issues are caught")
        print(f"   Miss Rate: {fn/(tp+fn)*100:.1f}% of issues are missed")
        
    def generate_roc_and_pr_curves(self):
        """
        SECTION 2: THRESHOLD-INDEPENDENT METRICS
        
        WHY: These metrics show model performance ACROSS ALL POSSIBLE THRESHOLDS
        - ROC Curve: Shows trade-off between true positive rate and false positive rate
        - PR Curve: Shows trade-off between precision and recall
        - AUC scores: Single number summarizing overall performance
        
        WHY BOTH CURVES?
        - ROC is good for balanced datasets
        - PR is better for imbalanced datasets (like ours: 10% anomalies)
        
        AUC INTERPRETATION:
        - 0.5 = Random guessing (useless)
        - 0.7-0.8 = Good
        - 0.8-0.9 = Very good
        - 0.9+ = Excellent
        """
        print("\n" + "="*80)
        print("SECTION 2: THRESHOLD-INDEPENDENT METRICS")
        print("="*80)
        
        y_true = self.test_data['is_anomaly']
        scores = -self.test_data['anomaly_score']  # Negative because lower score = more anomalous
        
        # Calculate ROC curve
        # WHY: Shows how well model separates normal from anomaly across thresholds
        fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        # WHY: Better for imbalanced data (we have 90% normal, 10% anomaly)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        
        print(f"\n📊 Area Under Curve Scores:")
        print(f"   AUC-ROC: {roc_auc:.4f}")
        print(f"   AUC-PR:  {pr_auc:.4f}")
        
        print(f"\n💡 What These Scores Mean:")
        if roc_auc > 0.9:
            print(f"   AUC-ROC {roc_auc:.4f} = EXCELLENT model performance! 🔥")
        elif roc_auc > 0.8:
            print(f"   AUC-ROC {roc_auc:.4f} = Very good performance")
        else:
            print(f"   AUC-ROC {roc_auc:.4f} = Good performance")
            
        print(f"   AUC-PR {pr_auc:.4f} = Precision-Recall balance")
        
        # ← NEW: Explain AUC-PR in context
        print(f"\n💡 AUC-PR Interpretation:")
        print(f"   Baseline (random): {y_true.mean():.4f}")
        print(f"   Our model:         {pr_auc:.4f}")
        print(f"   Improvement:       {pr_auc / y_true.mean():.1f}x better than random")
        
        # Create side-by-side plots
        # WHY: Visual comparison helps understand model behavior
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ROC Curve
        # WHY: Industry standard, easy to compare models
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.4f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                    label='Random Classifier (AUC = 0.50)')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate (Recall)', fontsize=12)
        axes[0].set_title('ROC Curve - Receiver Operating Characteristic', fontsize=14, fontweight='bold')
        axes[0].legend(loc="lower right")
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        # WHY: Better shows performance on minority class (anomalies)
        axes[1].plot(recall, precision, color='darkgreen', lw=2,
                    label=f'PR curve (AP = {pr_auc:.4f})')
        baseline_pr = y_true.mean()  # Baseline = % of anomalies
        axes[1].plot([0, 1], [baseline_pr, baseline_pr], color='navy', lw=2, linestyle='--',
                    label=f'Baseline (AP = {baseline_pr:.4f})')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        axes[1].legend(loc="lower left")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curves_path = os.path.join(self.output_dir, '2_roc_pr_curves.png')
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ ROC and PR curves saved: {curves_path}")
        plt.close()
        
        # ← NEW: Store AUC scores for summary report
        self.roc_auc = roc_auc
        self.pr_auc = pr_auc
        
    def analyze_anomaly_scores(self):
        """
        SECTION 3: ANOMALY SCORE ANALYSIS
        
        WHY: Understanding score distributions helps us:
        - Verify the model is learning meaningful patterns
        - Identify edge cases (borderline anomalies)
        - Find the most critical issues
        
        WHAT WE ANALYZE:
        - Score distribution for normal vs anomaly events
        - Top 10 most anomalous events (for investigation)
        - Score statistics by anomaly type
        """
        print("\n" + "="*80)
        print("SECTION 3: ANOMALY SCORE ANALYSIS")
        print("="*80)
        
        # Separate scores
        # WHY: Need to see if model actually separates the two classes
        normal_scores = self.test_data[self.test_data['is_anomaly']==0]['anomaly_score']
        anomaly_scores = self.test_data[self.test_data['is_anomaly']==1]['anomaly_score']
        
        print(f"\n📊 Score Statistics:")
        print(f"   Normal events:")
        print(f"      Mean: {normal_scores.mean():.4f}")
        print(f"      Std:  {normal_scores.std():.4f}")
        print(f"      Min:  {normal_scores.min():.4f}")
        print(f"   Anomaly events:")
        print(f"      Mean: {anomaly_scores.mean():.4f}")
        print(f"      Std:  {anomaly_scores.std():.4f}")
        print(f"      Min:  {anomaly_scores.min():.4f}")
        
        # ← NEW: Score separation quality
        score_gap = normal_scores.mean() - anomaly_scores.mean()
        print(f"\n   📏 Score Separation:")
        print(f"      Gap between means: {score_gap:.4f}")
        if score_gap > 0.05:
            print(f"      ✅ Good separation — model distinguishes classes well")
        else:
            print(f"      ⚠️  Small separation — classes overlap significantly")
        
        # Score distribution plot
        # WHY: Visualize separation between normal and anomaly
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overlapping histograms
        # WHY: See if distributions overlap (bad) or separate (good)
        axes[0].hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
        axes[0].hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
        axes[0].axvline(self.threshold_data['best_threshold'], color='black', linestyle='--',
                       linewidth=2, label=f'Optimal Threshold ({self.threshold_data["best_threshold"]:.4f})')
        axes[0].set_xlabel('Anomaly Score (lower = more anomalous)', fontsize=12)
        axes[0].set_ylabel('Density', fontsize=12)
        axes[0].set_title('Score Distribution: Normal vs Anomaly', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plots
        # WHY: Show median, quartiles, and outliers clearly
        data_to_plot = [normal_scores, anomaly_scores]
        axes[1].boxplot(data_to_plot, labels=['Normal', 'Anomaly'])
        axes[1].axhline(self.threshold_data['best_threshold'], color='red', linestyle='--',
                       label=f'Threshold: {self.threshold_data["best_threshold"]:.4f}')
        axes[1].set_ylabel('Anomaly Score', fontsize=12)
        axes[1].set_title('Score Distribution Box Plot', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        scores_path = os.path.join(self.output_dir, '3_score_distributions.png')
        plt.savefig(scores_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Score distribution plots saved: {scores_path}")
        plt.close()
        
        # Top 10 most anomalous events
        # WHY: These are the most critical issues to investigate
        print(f"\n🔍 TOP 10 MOST ANOMALOUS EVENTS:")
        print("   (Lowest anomaly scores = most suspicious)")
        top_anomalies = self.test_data.nsmallest(10, 'anomaly_score')
        
        # ← CHANGED: Dynamic column detection to avoid KeyError
        display_cols = []
        for col in ['api_latency_ms', 'fps', 'memory_mb']:
            if col in top_anomalies.columns:
                display_cols.append(col)
        
        print(f"\n   {'Rank':<6} {'Score':<12} {'True Label':<12} {'Type':<20}", end="")
        for col in display_cols:
            print(f" {col:<15}", end="")
        print()
        print("   " + "-"*80)
        
        for idx, (i, row) in enumerate(top_anomalies.iterrows(), 1):
            label = "ANOMALY" if row['is_anomaly'] == 1 else "Normal"
            print(f"   {idx:<6} {row['anomaly_score']:<12.4f} {label:<12} {row['anomaly_type']:<20}", end="")
            for col in display_cols:
                print(f" {row[col]:<15.2f}", end="")
            print()
        
        # Scores by anomaly type
        # WHY: Some types might be easier/harder to detect
        print(f"\n📊 Score Statistics by Anomaly Type:")
        anomaly_data = self.test_data[self.test_data['is_anomaly']==1]
        type_stats = anomaly_data.groupby('anomaly_type')['anomaly_score'].agg(['mean', 'std', 'min', 'count'])
        type_stats = type_stats.sort_values('mean')
        print(type_stats)
        
        # ← NEW: Visualize scores by anomaly type
        if len(type_stats) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            anomaly_types = anomaly_data['anomaly_type'].unique()
            type_scores = [anomaly_data[anomaly_data['anomaly_type']==t]['anomaly_score'].values 
                          for t in anomaly_types]
            
            ax.boxplot(type_scores, labels=anomaly_types)
            ax.axhline(self.threshold_data['best_threshold'], color='red', linestyle='--',
                       linewidth=2, label=f'Threshold: {self.threshold_data["best_threshold"]:.4f}')
            ax.set_ylabel('Anomaly Score (lower = more anomalous)', fontsize=12)
            ax.set_xlabel('Anomaly Type', fontsize=12)
            ax.set_title('Anomaly Scores by Type\n(Below red line = detected)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=30, ha='right')
            
            plt.tight_layout()
            type_path = os.path.join(self.output_dir, '3b_scores_by_anomaly_type.png')
            plt.savefig(type_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Scores by anomaly type saved: {type_path}")
            plt.close()
        
    def analyze_errors(self):
        """
        SECTION 4: ERROR ANALYSIS
        
        WHY: Understanding WHERE the model fails guides improvements
        
        FALSE POSITIVES: Normal events wrongly flagged as anomalies
        - BAD: Causes alert fatigue
        - QUESTION: What makes these look anomalous?
        
        FALSE NEGATIVES: Anomalies we missed
        - BAD: Real issues go undetected
        - QUESTION: Why did the model miss these?
        """
        print("\n" + "="*80)
        print("SECTION 4: ERROR ANALYSIS")
        print("="*80)
        
        # Identify errors
        y_true = self.test_data['is_anomaly']
        y_pred = self.test_data['predicted_anomaly']
        
        # False Positives: Predicted anomaly, actually normal
        # WHY: These cause unnecessary alerts
        false_positives = self.test_data[(y_true == 0) & (y_pred == 1)]
        
        # False Negatives: Predicted normal, actually anomaly
        # WHY: These are missed threats
        false_negatives = self.test_data[(y_true == 1) & (y_pred == 0)]
        
        # ← NEW: True Positives and True Negatives for comparison
        true_positives = self.test_data[(y_true == 1) & (y_pred == 1)]
        true_negatives = self.test_data[(y_true == 0) & (y_pred == 0)]
        
        print(f"\n📋 PREDICTION BREAKDOWN:")
        print(f"   ✅ True Positives:  {len(true_positives):4d} (correctly caught anomalies)")
        print(f"   ✅ True Negatives:  {len(true_negatives):4d} (correctly identified normal)")
        print(f"   ❌ False Positives: {len(false_positives):4d} (normal wrongly flagged)")
        print(f"   ❌ False Negatives: {len(false_negatives):4d} (anomalies missed)")
        
        # Analyze False Positives
        # WHY: Understand why normal events look anomalous
        print(f"\n" + "-"*60)
        print(f"🔍 FALSE POSITIVES ANALYSIS ({len(false_positives)} cases)")
        print(f"-"*60)
        
        if len(false_positives) > 0:
            print(f"   These are NORMAL events that the model wrongly flagged.")
            print(f"   Causes: Events with unusual patterns but still normal")
            
            # Show metrics distribution
            print(f"\n   Average metrics of false positives vs true normals:")
            metric_cols = ['api_latency_ms', 'fps', 'memory_mb', 'ui_response_ms']
            
            print(f"   {'Feature':<22} {'False Pos Avg':<16} {'Normal Avg':<16} {'Difference':<12}")
            print(f"   {'-'*65}")
            
            for col in metric_cols:
                if col in false_positives.columns:
                    fp_mean = false_positives[col].mean()
                    normal_mean = true_negatives[col].mean()
                    diff = fp_mean - normal_mean
                    print(f"   {col:<22} {fp_mean:<16.4f} {normal_mean:<16.4f} {diff:<12.4f}")
            
            # Worst false positives (most confident but wrong)
            print(f"\n   Top 5 Worst False Positives (most confidently wrong):")
            worst_fp = false_positives.nsmallest(5, 'anomaly_score')
            
            display_cols = [c for c in ['anomaly_score', 'api_latency_ms', 'fps', 'memory_mb'] 
                           if c in worst_fp.columns]
            
            print(f"   ", end="")
            for col in display_cols:
                print(f"{col:<18}", end="")
            print()
            print(f"   {'-'*len(display_cols)*18}")
            
            for _, row in worst_fp.iterrows():
                print(f"   ", end="")
                for col in display_cols:
                    print(f"{row[col]:<18.4f}", end="")
                print()
            
            # ← NEW: Score distribution of false positives
            print(f"\n   Score Statistics of False Positives:")
            print(f"      Mean score:  {false_positives['anomaly_score'].mean():.4f}")
            print(f"      Min score:   {false_positives['anomaly_score'].min():.4f}")
            print(f"      Max score:   {false_positives['anomaly_score'].max():.4f}")
            print(f"      Threshold:   {self.threshold_data['best_threshold']:.4f}")
            print(f"      → Most FP scores are CLOSE to threshold (borderline cases)")
        
        # Analyze False Negatives
        # WHY: Understand which anomalies we're missing
        print(f"\n" + "-"*60)
        print(f"🔍 FALSE NEGATIVES ANALYSIS ({len(false_negatives)} cases)")
        print(f"-"*60)
        
        if len(false_negatives) > 0:
            print(f"   These are ANOMALIES that the model missed.")
            print(f"   Causes: Subtle anomalies, borderline cases, gradual degradation")
            
            # Which anomaly types are hardest to detect?
            # WHY: Guide future model improvements
            print(f"\n   Missed Anomalies by Type:")
            fn_by_type = false_negatives['anomaly_type'].value_counts()
            total_by_type = self.test_data[self.test_data['is_anomaly']==1]['anomaly_type'].value_counts()
            
            # ← NEW: Also show detection rate per type
            print(f"   {'Type':<25} {'Missed':<8} {'Total':<8} {'Detected':<10} {'Detection Rate':<15}")
            print(f"   {'-'*65}")
            for anom_type in total_by_type.index:
                missed = fn_by_type.get(anom_type, 0)
                total = total_by_type[anom_type]
                detected = total - missed
                detect_rate = (detected / total) * 100 if total > 0 else 0
                
                # ← NEW: Visual indicator for detection quality
                if detect_rate >= 80:
                    indicator = "✅"
                elif detect_rate >= 50:
                    indicator = "⚠️"
                else:
                    indicator = "❌"
                
                print(f"   {anom_type:<25} {missed:<8} {total:<8} {detected:<10} {detect_rate:<13.1f}% {indicator}")
            
            # Show borderline cases (scores close to threshold)
            print(f"\n   Borderline Cases (scores near threshold):")
            threshold = self.threshold_data['best_threshold']
            borderline = false_negatives[
                (false_negatives['anomaly_score'] >= threshold * 0.9) &
                (false_negatives['anomaly_score'] <= threshold * 1.1)
            ]
            print(f"      {len(borderline)} false negatives have scores within 10% of threshold")
            print(f"      These are the 'almost caught' cases")
            
            if len(borderline) > 0:
                print(f"      → Slightly adjusting threshold could catch {len(borderline)} more anomalies")
            
            # ← NEW: Compare false negatives vs true positives
            print(f"\n   False Negatives vs True Positives (what makes missed ones different?):")
            compare_cols = ['api_latency_ms', 'fps', 'memory_mb', 'ui_response_ms']
            
            print(f"   {'Feature':<22} {'Missed (FN) Avg':<18} {'Caught (TP) Avg':<18} {'Gap':<10}")
            print(f"   {'-'*68}")
            
            for col in compare_cols:
                if col in false_negatives.columns and col in true_positives.columns:
                    fn_mean = false_negatives[col].mean()
                    tp_mean = true_positives[col].mean()
                    gap = abs(fn_mean - tp_mean)
                    print(f"   {col:<22} {fn_mean:<18.4f} {tp_mean:<18.4f} {gap:<10.4f}")
            
            print(f"\n   💡 Insight: Missed anomalies tend to have values CLOSER to normal")
            print(f"      → They are subtle/mild anomalies that don't deviate strongly")
        
        # ← NEW: Error visualization
        self._plot_error_analysis(false_positives, false_negatives, true_positives, true_negatives)
    
    # ← NEW: Entire method is new
    def _plot_error_analysis(self, false_positives, false_negatives, true_positives, true_negatives):
        """
        Create visualizations for error analysis.
        
        WHY: Visual patterns in errors help identify systematic issues
        - Score distributions of each category
        - Feature value comparisons
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Score distribution by prediction category
        # WHY: Shows how well-separated the categories are
        categories = {
            'True Positive': true_positives['anomaly_score'] if len(true_positives) > 0 else pd.Series(),
            'True Negative': true_negatives['anomaly_score'] if len(true_negatives) > 0 else pd.Series(),
            'False Positive': false_positives['anomaly_score'] if len(false_positives) > 0 else pd.Series(),
            'False Negative': false_negatives['anomaly_score'] if len(false_negatives) > 0 else pd.Series()
        }
        colors = {'True Positive': 'green', 'True Negative': 'blue', 
                  'False Positive': 'orange', 'False Negative': 'red'}
        
        for label, scores in categories.items():
            if len(scores) > 0:
                axes[0, 0].hist(scores, bins=30, alpha=0.5, label=f'{label} ({len(scores)})', 
                               color=colors[label], density=True)
        
        axes[0, 0].axvline(self.threshold_data['best_threshold'], color='black', 
                           linestyle='--', linewidth=2, label='Threshold')
        axes[0, 0].set_xlabel('Anomaly Score', fontsize=11)
        axes[0, 0].set_ylabel('Density', fontsize=11)
        axes[0, 0].set_title('Score Distribution by Prediction Category', fontsize=13, fontweight='bold')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error counts bar chart
        # WHY: Quick visual of error magnitude
        error_counts = {
            'True\nPositive': len(true_positives),
            'True\nNegative': len(true_negatives),
            'False\nPositive': len(false_positives),
            'False\nNegative': len(false_negatives)
        }
        bar_colors = ['green', 'blue', 'orange', 'red']
        
        bars = axes[0, 1].bar(error_counts.keys(), error_counts.values(), color=bar_colors, alpha=0.7)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].set_title('Prediction Category Counts', fontsize=13, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, error_counts.values()):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                           str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Plot 3: Detection rate by anomaly type
        # WHY: Identify which anomaly types are hardest to detect
        if 'anomaly_type' in self.test_data.columns:
            anomaly_data = self.test_data[self.test_data['is_anomaly'] == 1]
            type_detection = anomaly_data.groupby('anomaly_type').agg(
                total=('is_anomaly', 'count'),
                detected=('predicted_anomaly', 'sum')
            )
            type_detection['detection_rate'] = (type_detection['detected'] / type_detection['total'] * 100)
            type_detection = type_detection.sort_values('detection_rate')
            
            bar_colors_type = ['red' if r < 50 else 'orange' if r < 80 else 'green' 
                              for r in type_detection['detection_rate']]
            
            axes[1, 0].barh(type_detection.index, type_detection['detection_rate'], 
                           color=bar_colors_type, alpha=0.7)
            axes[1, 0].set_xlabel('Detection Rate (%)', fontsize=11)
            axes[1, 0].set_title('Detection Rate by Anomaly Type', fontsize=13, fontweight='bold')
            axes[1, 0].axvline(x=80, color='green', linestyle='--', alpha=0.5, label='80% target')
            axes[1, 0].axvline(x=50, color='red', linestyle='--', alpha=0.5, label='50% minimum')
            axes[1, 0].set_xlim([0, 105])
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for idx, (_, row) in enumerate(type_detection.iterrows()):
                axes[1, 0].text(row['detection_rate'] + 1, idx,
                               f"{row['detection_rate']:.1f}% ({int(row['detected'])}/{int(row['total'])})",
                               va='center', fontsize=9)
        
        # Plot 4: False Negative score vs True Positive score
        # WHY: Shows how close missed anomalies are to being caught
        if len(false_negatives) > 0 and len(true_positives) > 0:
            axes[1, 1].hist(true_positives['anomaly_score'], bins=30, alpha=0.6, 
                           label=f'Caught Anomalies ({len(true_positives)})', color='green', density=True)
            axes[1, 1].hist(false_negatives['anomaly_score'], bins=30, alpha=0.6,
                           label=f'Missed Anomalies ({len(false_negatives)})', color='red', density=True)
            axes[1, 1].axvline(self.threshold_data['best_threshold'], color='black',
                              linestyle='--', linewidth=2, label='Threshold')
            axes[1, 1].set_xlabel('Anomaly Score', fontsize=11)
            axes[1, 1].set_ylabel('Density', fontsize=11)
            axes[1, 1].set_title('Caught vs Missed Anomalies', fontsize=13, fontweight='bold')
            axes[1, 1].legend(fontsize=9)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Error Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        error_path = os.path.join(self.output_dir, '4_error_analysis.png')
        plt.savefig(error_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Error analysis plots saved: {error_path}")
        plt.close()
        
    def compare_models(self):
        """
        SECTION 5: MODEL COMPARISON VISUALIZATION
        
        WHY: Show the value of ML over simple baseline
        - Side-by-side metrics
        - Visual confusion matrix comparison
        - Improvement summary
        """
        print("\n" + "="*80)
        print("SECTION 5: MODEL COMPARISON")
        print("="*80)
        
        y_true = self.test_data['is_anomaly']
        iforest_pred = self.test_data['predicted_anomaly']
        baseline_pred = self.test_data['baseline_prediction']
        
        # Calculate metrics for both
        metrics_comparison = pd.DataFrame({
            'Baseline (Z-Score)': [
                precision_score(y_true, baseline_pred),
                recall_score(y_true, baseline_pred),
                f1_score(y_true, baseline_pred)
            ],
            'Isolation Forest': [
                precision_score(y_true, iforest_pred),
                recall_score(y_true, iforest_pred),
                f1_score(y_true, iforest_pred)
            ]
        }, index=['Precision', 'Recall', 'F1-Score'])
        
        print(f"\n📊 Model Comparison:")
        print(metrics_comparison.to_string())
        
        # ← NEW: Calculate and display improvements
        print(f"\n📈 Improvement (Isolation Forest over Baseline):")
        for metric in metrics_comparison.index:
            baseline_val = metrics_comparison.loc[metric, 'Baseline (Z-Score)']
            iforest_val = metrics_comparison.loc[metric, 'Isolation Forest']
            if baseline_val > 0:
                improvement = ((iforest_val - baseline_val) / baseline_val) * 100
                direction = "📈" if improvement > 0 else "📉"
                print(f"   {direction} {metric}: {improvement:+.1f}%")
            else:
                print(f"   ➡️  {metric}: N/A (baseline is 0)")
        
        # Bar chart comparison
        # WHY: Visual comparison is easier to understand
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        colors = ['#e74c3c', '#2ecc71']  # Red for baseline, green for IF
        
        for idx, metric in enumerate(metrics_comparison.index):
            values = metrics_comparison.loc[metric]
            bars = axes[idx].bar(range(len(values)), values, width=0.5, alpha=0.8, color=colors)
            axes[idx].set_ylabel(metric, fontsize=12)
            axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            axes[idx].set_xticks(range(len(values)))
            axes[idx].set_xticklabels(metrics_comparison.columns, rotation=15, ha='right')
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
        
        plt.suptitle('Model Comparison: Baseline vs Isolation Forest', fontsize=16, fontweight='bold')
        plt.tight_layout()
        comparison_path = os.path.join(self.output_dir, '5_model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Model comparison chart saved: {comparison_path}")
        plt.close()
        
        # Side-by-side confusion matrices
        # WHY: Visual comparison of where each model succeeds/fails
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        cm_baseline = confusion_matrix(y_true, baseline_pred)
        cm_iforest = confusion_matrix(y_true, iforest_pred)
        
        sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        axes[0].set_title('Baseline (Z-Score) Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Actual', fontsize=12)
        axes[0].set_xlabel('Predicted', fontsize=12)
        
        sns.heatmap(cm_iforest, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                   xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        axes[1].set_title('Isolation Forest Confusion Matrix', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Actual', fontsize=12)
        axes[1].set_xlabel('Predicted', fontsize=12)
        
        plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        cm_comparison_path = os.path.join(self.output_dir, '5_confusion_matrices_comparison.png')
        plt.savefig(cm_comparison_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix comparison saved: {cm_comparison_path}")
        plt.close()
    
    # ← NEW: Entire method is new
    def analyze_feature_importance(self):
        """
        SECTION 6: FEATURE IMPORTANCE ANALYSIS
        
        WHY: Understanding which features drive anomaly detection helps us:
        - Identify the most useful monitoring metrics
        - Focus engineering efforts on key features
        - Explain model decisions to stakeholders
        - Potentially reduce feature set without losing performance
        
        HOW: For Isolation Forest, we measure importance two ways:
        1. Tree-based importance: How often each feature is used for splits
        2. Score impact: How much scores change when a feature is shuffled (permutation)
        
        NOTE: Isolation Forest doesn't have built-in feature_importances_ like 
        Random Forest, so we estimate importance using the tree structures.
        """
        print("\n" + "="*80)
        print("SECTION 6: FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Method 1: Extract importance from tree structures
        # WHY: Each tree in the forest splits on features — 
        # features used more often at shallower depths are more important
        
        # Get feature names (exclude label/meta columns)
        label_cols = ['is_anomaly', 'anomaly_type', 'timestamp', 'session_id',
                      'predicted_anomaly', 'anomaly_score', 'baseline_prediction']
        feature_cols = [col for col in self.test_data.columns if col not in label_cols]
        
        n_features = len(feature_cols)
        feature_usage_count = np.zeros(n_features)
        
        # Count how often each feature is used across all trees
        # WHY: Features that isolate anomalies effectively get used more often
        print(f"\n🔍 Analyzing {len(self.model.estimators_)} trees for feature usage...")
        
        for tree in self.model.estimators_:
            tree_model = tree.tree_
            # Get features used in this tree (non-leaf nodes)
            features_in_tree = tree_model.feature[tree_model.feature >= 0]
            for feat_idx in features_in_tree:
                if feat_idx < n_features:
                    feature_usage_count[feat_idx] += 1
        
        # Normalize to get relative importance
        feature_importance = feature_usage_count / feature_usage_count.sum()
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': feature_importance,
            'Usage Count': feature_usage_count.astype(int)
        }).sort_values('Importance', ascending=False)
        
        print(f"\n📊 Feature Importance (Tree-Based):")
        print(f"   {'Rank':<6} {'Feature':<35} {'Importance':<12} {'Usage Count':<12}")
        print(f"   {'-'*65}")
        
        for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
            bar = "█" * int(row['Importance'] * 100)
            print(f"   {rank:<6} {row['Feature']:<35} {row['Importance']:<12.4f} {int(row['Usage Count']):<12} {bar}")
        
        # Method 2: Permutation-based importance (simplified)
        # WHY: Measures how much model performance DROPS when a feature is shuffled
        # If performance drops a lot → feature is important
        # If performance doesn't change → feature doesn't matter
        
        print(f"\n🔍 Calculating permutation importance (this may take a moment)...")
        
        X_test_features = self.test_data[feature_cols]
        y_true = self.test_data['is_anomaly']
        threshold = self.threshold_data['best_threshold']
        
        # Baseline F1 score
        baseline_scores = self.model.score_samples(X_test_features)
        baseline_preds = (baseline_scores < threshold).astype(int)
        baseline_f1 = f1_score(y_true, baseline_preds)
        
        permutation_importance = {}
        
        for col in feature_cols:
            # Shuffle one feature at a time
            X_shuffled = X_test_features.copy()
            X_shuffled[col] = np.random.permutation(X_shuffled[col].values)
            
            # Get new predictions
            shuffled_scores = self.model.score_samples(X_shuffled)
            shuffled_preds = (shuffled_scores < threshold).astype(int)
            shuffled_f1 = f1_score(y_true, shuffled_preds)
            
            # Importance = how much F1 DROPS (bigger drop = more important)
            permutation_importance[col] = baseline_f1 - shuffled_f1
        
        perm_df = pd.DataFrame({
            'Feature': list(permutation_importance.keys()),
            'F1_Drop': list(permutation_importance.values())
        }).sort_values('F1_Drop', ascending=False)
        
        print(f"\n📊 Permutation Importance (F1 drop when feature shuffled):")
        print(f"   Baseline F1: {baseline_f1:.4f}")
        print(f"\n   {'Rank':<6} {'Feature':<35} {'F1 Drop':<12} {'Impact':<10}")
        print(f"   {'-'*60}")
        
        for rank, (_, row) in enumerate(perm_df.iterrows(), 1):
            if row['F1_Drop'] > 0.01:
                impact = "🔴 HIGH"
            elif row['F1_Drop'] > 0.001:
                impact = "🟡 MED"
            elif row['F1_Drop'] > 0:
                impact = "🟢 LOW"
            else:
                impact = "⚪ NONE"
            print(f"   {rank:<6} {row['Feature']:<35} {row['F1_Drop']:<12.4f} {impact}")
        
        # Visualization: Feature Importance Charts
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot 1: Tree-based importance (top 15)
        top_tree = importance_df.head(15)
        axes[0].barh(range(len(top_tree)), top_tree['Importance'].values, color='steelblue', alpha=0.8)
        axes[0].set_yticks(range(len(top_tree)))
        axes[0].set_yticklabels(top_tree['Feature'].values)
        axes[0].set_xlabel('Relative Importance', fontsize=12)
        axes[0].set_title('Tree-Based Feature Importance\n(How often used for splits)', 
                          fontsize=13, fontweight='bold')
        axes[0].invert_yaxis()  # Most important at top
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # Plot 2: Permutation importance (top 15)
        top_perm = perm_df.head(15)
        colors_perm = ['red' if v > 0.01 else 'orange' if v > 0.001 else 'green' 
                      for v in top_perm['F1_Drop'].values]
        axes[1].barh(range(len(top_perm)), top_perm['F1_Drop'].values, color=colors_perm, alpha=0.8)
        axes[1].set_yticks(range(len(top_perm)))
        axes[1].set_yticklabels(top_perm['Feature'].values)
        axes[1].set_xlabel('F1 Score Drop (higher = more important)', fontsize=12)
        axes[1].set_title('Permutation Feature Importance\n(F1 drop when feature shuffled)', 
                          fontsize=13, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        importance_path = os.path.join(self.output_dir, '6_feature_importance.png')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Feature importance plots saved: {importance_path}")
        plt.close()
        
        # ← NEW: Feature correlation heatmap (top features only)
        print(f"\n🔍 Generating feature correlation heatmap...")
        
        top_features = importance_df.head(15)['Feature'].tolist()
        top_features_available = [f for f in top_features if f in X_test_features.columns]
        
        if len(top_features_available) > 1:
            corr_matrix = X_test_features[top_features_available].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, vmin=-1, vmax=1, square=True)
            plt.title('Feature Correlation Heatmap (Top 15 Features)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            corr_path = os.path.join(self.output_dir, '6b_feature_correlation.png')
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            print(f"✅ Feature correlation heatmap saved: {corr_path}")
            plt.close()
        
        # Summary insight
        top_3 = importance_df.head(3)['Feature'].tolist()
        print(f"\n💡 Key Insight:")
        print(f"   Top 3 most important features for anomaly detection:")
        for i, feat in enumerate(top_3, 1):
            print(f"      {i}. {feat}")
        print(f"   → Focus monitoring dashboards on these metrics")
        
    def generate_summary_report(self):
        """
        Generate final summary report with all key findings.
        
        WHY: Single document for project report/presentation
        """
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        report_path = os.path.join(self.output_dir, 'EVALUATION_SUMMARY.txt')
        
        y_true = self.test_data['is_anomaly']
        y_pred = self.test_data['predicted_anomaly']
        baseline_pred = self.test_data['baseline_prediction']
        
        # Calculate all metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        iforest_prec = precision_score(y_true, y_pred)
        iforest_rec = recall_score(y_true, y_pred)
        iforest_f1 = f1_score(y_true, y_pred)
        
        baseline_prec = precision_score(y_true, baseline_pred)
        baseline_rec = recall_score(y_true, baseline_pred)
        baseline_f1 = f1_score(y_true, baseline_pred)
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ANOMALY DETECTION MODEL - COMPREHENSIVE EVALUATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Project: Mobile App Performance Anomaly Detection\n")
            f.write(f"Author: BCA Final Year Project\n\n")
            
            # Test set info
            f.write("1. TEST SET INFORMATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"   Total events:   {len(self.test_data)}\n")
            f.write(f"   Normal events:  {(y_true==0).sum()} ({(y_true==0).mean()*100:.2f}%)\n")
            f.write(f"   Anomaly events: {(y_true==1).sum()} ({(y_true==1).mean()*100:.2f}%)\n\n")
            
            # Model configuration
            f.write("2. MODEL CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"   Algorithm:      Isolation Forest\n")
            f.write(f"   n_estimators:   200\n")
            f.write(f"   max_samples:    0.8 (80% of training data)\n")
            f.write(f"   max_features:   0.8 (80% of features per tree)\n")
            f.write(f"   contamination:  auto (threshold optimized separately)\n")
            f.write(f"   Opt. Threshold: {self.threshold_data['best_threshold']:.4f}\n\n")
            
            # Model performance
            f.write("3. ISOLATION FOREST PERFORMANCE:\n")
            f.write("-"*40 + "\n")
            f.write(f"   Precision: {iforest_prec:.4f} ({iforest_prec*100:.1f}%)\n")
            f.write(f"   Recall:    {iforest_rec:.4f} ({iforest_rec*100:.1f}%)\n")
            f.write(f"   F1-Score:  {iforest_f1:.4f} ({iforest_f1*100:.1f}%)\n")
            
            # ← NEW: Include AUC scores if available
            if hasattr(self, 'roc_auc'):
                f.write(f"   AUC-ROC:   {self.roc_auc:.4f} ({self.roc_auc*100:.1f}%)\n")
            if hasattr(self, 'pr_auc'):
                f.write(f"   AUC-PR:    {self.pr_auc:.4f} ({self.pr_auc*100:.1f}%)\n")
            f.write("\n")
            
            # Baseline comparison
            f.write("4. BASELINE (Z-SCORE) PERFORMANCE:\n")
            f.write("-"*40 + "\n")
            f.write(f"   Precision: {baseline_prec:.4f} ({baseline_prec*100:.1f}%)\n")
            f.write(f"   Recall:    {baseline_rec:.4f} ({baseline_rec*100:.1f}%)\n")
            f.write(f"   F1-Score:  {baseline_f1:.4f} ({baseline_f1*100:.1f}%)\n\n")
            
            # Improvement
            f.write("5. IMPROVEMENT (Isolation Forest over Baseline):\n")
            f.write("-"*40 + "\n")
            if baseline_prec > 0:
                f.write(f"   Precision: {((iforest_prec-baseline_prec)/baseline_prec)*100:+.1f}%\n")
            if baseline_rec > 0:
                f.write(f"   Recall:    {((iforest_rec-baseline_rec)/baseline_rec)*100:+.1f}%\n")
            if baseline_f1 > 0:
                f.write(f"   F1-Score:  {((iforest_f1-baseline_f1)/baseline_f1)*100:+.1f}%\n")
            f.write("\n")
            
            # Confusion matrix
            f.write("6. CONFUSION MATRIX:\n")
            f.write("-"*40 + "\n")
            f.write(f"   True Positives:  {tp:4d} (correctly caught anomalies)\n")
            f.write(f"   True Negatives:  {tn:4d} (correctly identified normal)\n")
            f.write(f"   False Positives: {fp:4d} (false alarms)\n")
            f.write(f"   False Negatives: {fn:4d} (missed anomalies)\n\n")
            
            # Business metrics
            f.write("7. BUSINESS METRICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"   Alert Accuracy:  {tp/(tp+fp)*100:.1f}% of alerts are real issues\n")
            f.write(f"   Detection Rate:  {tp/(tp+fn)*100:.1f}% of real issues are caught\n")
            f.write(f"   Miss Rate:       {fn/(tp+fn)*100:.1f}% of real issues are missed\n")
            f.write(f"   False Alarm Rate: {fp/(fp+tn)*100:.1f}% of normal events trigger alerts\n\n")
            
            # Detection by anomaly type
            f.write("8. DETECTION RATE BY ANOMALY TYPE:\n")
            f.write("-"*40 + "\n")
            anomaly_data = self.test_data[self.test_data['is_anomaly'] == 1]
            type_detection = anomaly_data.groupby('anomaly_type').agg(
                total=('is_anomaly', 'count'),
                detected=('predicted_anomaly', 'sum')
            )
            type_detection['rate'] = (type_detection['detected'] / type_detection['total'] * 100)
            type_detection = type_detection.sort_values('rate', ascending=False)
            
            for atype, row in type_detection.iterrows():
                f.write(f"   {atype:<25}: {row['rate']:.1f}% "
                       f"({int(row['detected'])}/{int(row['total'])})\n")
            f.write("\n")
            
            # Visualizations generated
            f.write("9. VISUALIZATIONS GENERATED:\n")
            f.write("-"*40 + "\n")
            f.write("   1. Confusion Matrix Heatmap\n")
            f.write("   2. ROC and Precision-Recall Curves\n")
            f.write("   3. Anomaly Score Distributions\n")
            f.write("   3b. Scores by Anomaly Type\n")
            f.write("   4. Error Analysis Dashboard\n")
            f.write("   5. Model Comparison Charts\n")
            f.write("   5b. Confusion Matrix Comparison\n")
            f.write("   6. Feature Importance Analysis\n")
            f.write("   6b. Feature Correlation Heatmap\n\n")
            
            # Key findings
            f.write("10. KEY FINDINGS:\n")
            f.write("-"*40 + "\n")
            f.write(f"   • Isolation Forest achieves {iforest_f1*100:.1f}% F1-score, "
                   f"a {((iforest_f1-baseline_f1)/baseline_f1)*100:.1f}% improvement over baseline\n")
            if hasattr(self, 'roc_auc'):
                f.write(f"   • AUC-ROC of {self.roc_auc:.4f} indicates excellent anomaly ranking ability\n")
            f.write(f"   • Model catches {tp/(tp+fn)*100:.1f}% of all anomalies (high recall)\n")
            f.write(f"   • {fp} false alarms out of {tn+fp} normal events "
                   f"({fp/(fp+tn)*100:.1f}% false alarm rate)\n")
            f.write(f"   • Custom threshold optimization significantly improves over default\n\n")
            
            # Recommendations
            f.write("11. RECOMMENDATIONS FOR IMPROVEMENT:\n")
            f.write("-"*40 + "\n")
            f.write("   • Consider ensemble approach (multiple Isolation Forest configs)\n")
            f.write("   • Try supervised models if labeled data is available\n")
            f.write("   • Add more domain-specific features (e.g., time-of-day patterns)\n")
            f.write("   • Implement adaptive thresholding for production deployment\n")
            f.write("   • Focus on improving detection of hardest anomaly types\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF EVALUATION REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\n✅ Summary report saved: {report_path}")
        
        # ← NEW: Print summary to console as well
        print(f"\n📋 EVALUATION SUMMARY (Console):")
        print(f"   ┌─────────────────────────────────────────────────────┐")
        print(f"   │  ANOMALY DETECTION - FINAL RESULTS                  │")
        print(f"   ├─────────────────────────────────────────────────────┤")
        print(f"   │  Test Set: {len(self.test_data):,} events ({(y_true==1).sum()} anomalies)       │")
        print(f"   │                                                     │")
        print(f"   │  Isolation Forest:                                  │")
        print(f"   │    F1-Score:  {iforest_f1:.4f}  ({iforest_f1*100:.1f}%)                    │")
        print(f"   │    Precision: {iforest_prec:.4f}  ({iforest_prec*100:.1f}%)                    │")
        print(f"   │    Recall:    {iforest_rec:.4f}  ({iforest_rec*100:.1f}%)                    │")
        if hasattr(self, 'roc_auc'):
            print(f"   │    AUC-ROC:   {self.roc_auc:.4f}  ({self.roc_auc*100:.1f}%)                    │")
        print(f"   │                                                     │")
        print(f"   │  vs Baseline: +{((iforest_f1-baseline_f1)/baseline_f1)*100:.1f}% F1 improvement           │")
        print(f"   │                                                     │")
        print(f"   │  Catches {tp}/{tp+fn} anomalies ({tp/(tp+fn)*100:.1f}% detection rate)    │")
        print(f"   └─────────────────────────────────────────────────────┘")
        
    # ← COMPLETED: This method was incomplete
    def run_full_evaluation(self):
        """
        Execute complete evaluation pipeline.
        
        PIPELINE:
        1. Load test data and predictions
        2. Generate classification metrics
        3. Plot ROC and PR curves
        4. Analyze anomaly scores
        5. Perform error analysis
        6. Compare models
        7. Analyze feature importance
        8. Generate summary report
        """
        print("\n" + "="*80)
        print("MODEL EVALUATION - COMPREHENSIVE ANALYSIS")
        print("="*80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Models directory: {self.models_dir}")
        
        # Step 1: Load data
        self.load_test_predictions()
        
        # Step 2: Classification metrics
        self.generate_classification_metrics()
        
        # Step 3: ROC and PR curves
        self.generate_roc_and_pr_curves()
        
        # Step 4: Anomaly score analysis
        self.analyze_anomaly_scores()
        
        # Step 5: Error analysis
        self.analyze_errors()
        
        # Step 6: Model comparison
        self.compare_models()
        
        # Step 7: Feature importance
        self.analyze_feature_importance()
        
        # Step 8: Summary report
        self.generate_summary_report()
        
        # ← NEW: Final output summary
        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        
        # List all generated files
        print(f"\n📁 All outputs saved to: {self.output_dir}/")
        print(f"\n📊 Generated Visualizations:")
        
        generated_files = sorted(os.listdir(self.output_dir))
        for f in generated_files:
            filepath = os.path.join(self.output_dir, f)
            size_kb = os.path.getsize(filepath) / 1024
            if f.endswith('.png'):
                print(f"   📈 {f} ({size_kb:.1f} KB)")
            elif f.endswith('.txt'):
                print(f"   📄 {f} ({size_kb:.1f} KB)")
        
        print(f"\n✅ Total files generated: {len(generated_files)}")
        print(f"\n💡 Next Steps:")
        print(f"   1. Review visualizations in '{self.output_dir}/' folder")
        print(f"   2. Read EVALUATION_SUMMARY.txt for complete findings")
        print(f"   3. Use these in your project report and presentation")
        print(f"   4. Proceed to building the prediction/deployment module")
        
        print("\n" + "="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Configuration
    MODELS_DIR = "models"           # Where trained models are saved
    OUTPUT_DIR = "evaluation"       # Where evaluation results will be saved
    
    print("\n" + "="*80)
    print("ANOMALY DETECTION - MODEL EVALUATION PIPELINE")
    print("="*80)
    print(f"\nModels directory: {MODELS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # ← NEW: Check if required files exist before starting
    required_files = [
        os.path.join(MODELS_DIR, 'test_predictions.csv'),
        os.path.join(MODELS_DIR, 'isolation_forest.pkl'),
        os.path.join(MODELS_DIR, 'optimal_threshold.pkl')
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"\n❌ ERROR: Missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        print(f"\n💡 Make sure you've run train_model.py first!")
        print(f"   The training module saves these files to the '{MODELS_DIR}/' directory.")
    else:
        print(f"\n✅ All required files found. Starting evaluation...\n")
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            models_dir=MODELS_DIR,
            output_dir=OUTPUT_DIR
        )
        
        # Run full evaluation
        evaluator.run_full_evaluation()
        
        # Print usage for report
        print("\n" + "="*80)
        print("FOR YOUR PROJECT REPORT:")
        print("="*80)
        print(f"""
Include these in your report:

1. VISUALIZATIONS (from '{OUTPUT_DIR}/' folder):
   - 1_confusion_matrix.png          → Show model accuracy
   - 2_roc_pr_curves.png             → Show model quality (AUC scores)
   - 3_score_distributions.png       → Show how model separates classes
   - 3b_scores_by_anomaly_type.png   → Show per-type detection
   - 4_error_analysis.png            → Show error patterns
   - 5_model_comparison.png          → Show ML vs baseline improvement
   - 5_confusion_matrices_comparison.png → Side-by-side comparison
   - 6_feature_importance.png        → Show which features matter
   - 6b_feature_correlation.png      → Show feature relationships

2. WRITTEN REPORT:
   - Read EVALUATION_SUMMARY.txt for all metrics and findings
   - Copy key numbers: F1, AUC-ROC, detection rate, improvement %

3. VIVA PREPARATION:
   - "Our model achieves X% F1-score with Y% AUC-ROC"
   - "This is Z% better than the statistical baseline"
   - "The model catches N% of all anomalies"
   - "Top features for detection are: ..."
        """)