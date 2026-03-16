"""
preprocess.py - Mobile App Performance Telemetry Data Preprocessing

This module prepares the generated telemetry data for Isolation Forest training.
It handles feature selection, train/test splitting, and scaling while preventing
data leakage and maintaining temporal integrity.

Author: BCA Final Year Project
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
from datetime import datetime

class TelemetryPreprocessor:
    """
    Preprocesses mobile app telemetry data for anomaly detection.
    
    Key Principles:
    1. NO label leakage - labels removed from training features
    2. Chronological train/test split - respects time-series nature
    3. Feature scaling - StandardScaler for Isolation Forest
    4. Categorical encoding - Convert app_version, network_type
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.preprocessing_stats = {}
        
    def load_data(self, filepath):
        """
        Load telemetry data from CSV.
        
        WHY: Centralized data loading with validation
        """
        print("="*80)
        print("STEP 1: LOADING DATA")
        print("="*80)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"\n✅ Loaded {len(df):,} events from {filepath}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Store original data info
        self.preprocessing_stats['total_events'] = len(df)
        self.preprocessing_stats['date_range'] = (df['timestamp'].min(), df['timestamp'].max())
        
        return df
    
    def separate_features_labels(self, df):
        """
        Separate features from ground truth labels.
        
        WHY: Prevent label leakage during training
        
        CRITICAL: The model must NEVER see is_anomaly or anomaly_type during training.
        These are only for evaluation after predictions are made.
        """
        print("\n" + "="*80)
        print("STEP 2: SEPARATING FEATURES AND LABELS")
        print("="*80)
        
        # Define label columns (DO NOT USE IN TRAINING)
        label_columns = ['is_anomaly', 'anomaly_type', 'session_health']
        
        # Define metadata columns (useful for grouping but not training)
        metadata_columns = ['timestamp', 'session_id']
        
        # Extract labels
        y = df['is_anomaly'].copy()
        anomaly_types = df['anomaly_type'].copy()
        
        # Extract metadata for later analysis
        metadata = df[metadata_columns].copy()
        
        # Feature columns = everything except labels and metadata
        feature_columns = [col for col in df.columns 
                          if col not in label_columns + metadata_columns]
        
        X = df[feature_columns].copy()
        
        print(f"\n✅ Features extracted: {len(feature_columns)} columns")
        print(f"   Labels extracted: {len(label_columns)} columns")
        print(f"   Metadata extracted: {len(metadata_columns)} columns")
        
        print(f"\n📊 Anomaly Distribution:")
        print(f"   Normal events: {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%)")
        print(f"   Anomaly events: {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")
        
        print(f"\n📋 Feature Categories:")
        base_features = ['app_version', 'screen_name', 'api_latency_ms', 'ui_response_ms', 
                        'fps', 'memory_mb', 'error_code', 'network_type']
        rolling_features = [col for col in feature_columns if 'rolling' in col]
        zscore_features = [col for col in feature_columns if 'zscore' in col]
        rate_features = [col for col in feature_columns if 'rate' in col or 'growth' in col]
        other_features = [col for col in feature_columns 
                         if col not in base_features + rolling_features + zscore_features + rate_features]
        
        print(f"   Base metrics: {len(base_features)}")
        print(f"   Rolling statistics: {len(rolling_features)}")
        print(f"   Z-score features: {len(zscore_features)}")
        print(f"   Rate/growth features: {len(rate_features)}")
        print(f"   Other features: {len(other_features)}")
        
        self.feature_names = feature_columns
        self.preprocessing_stats['n_features'] = len(feature_columns)
        
        return X, y, anomaly_types, metadata
    
    def encode_categorical_features(self, X):
        """
        Encode categorical features (app_version, screen_name, network_type).
        
        WHY: Isolation Forest requires numeric features.
        
        METHOD: LabelEncoder (converts categories to integers 0, 1, 2, ...)
        Alternative: One-Hot Encoding would create too many sparse features.
        """
        print("\n" + "="*80)
        print("STEP 3: ENCODING CATEGORICAL FEATURES")
        print("="*80)
        
        categorical_cols = ['app_version', 'screen_name', 'network_type']
        
        for col in categorical_cols:
            if col in X.columns:
                print(f"\n🔄 Encoding '{col}':")
                print(f"   Unique values: {X[col].nunique()}")
                print(f"   Categories: {sorted(X[col].unique())}")
                
                # Initialize and fit encoder
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                
                # Store encoder for later use (inverse transform if needed)
                self.label_encoders[col] = le
                
                print(f"   ✅ Encoded to: {sorted(X[col].unique())}")
        
        return X
    
    def chronological_split(self, X, y, anomaly_types, metadata, test_size=0.2):
        """
        Split data chronologically (not randomly).
        
        WHY: Time-series data has temporal dependencies.
        Random split would leak future information into training.
        
        APPROACH:
        - First 80% of events = Training set
        - Last 20% of events = Test set
        
        This simulates real deployment: train on past data, predict future.
        """
        print("\n" + "="*80)
        print("STEP 4: CHRONOLOGICAL TRAIN/TEST SPLIT")
        print("="*80)
        
        print(f"\n⚠️  Using CHRONOLOGICAL split (not random)")
        print(f"   Reason: Preserves temporal order and prevents data leakage")
        
        # Sort by timestamp (already sorted, but ensure it)
        sort_idx = metadata.sort_values('timestamp').index
        X_sorted = X.loc[sort_idx]
        y_sorted = y.loc[sort_idx]
        anomaly_types_sorted = anomaly_types.loc[sort_idx]
        metadata_sorted = metadata.loc[sort_idx]
        
        # Calculate split point
        split_idx = int(len(X_sorted) * (1 - test_size))
        
        # Split
        X_train = X_sorted.iloc[:split_idx].reset_index(drop=True)
        X_test = X_sorted.iloc[split_idx:].reset_index(drop=True)
        y_train = y_sorted.iloc[:split_idx].reset_index(drop=True)
        y_test = y_sorted.iloc[split_idx:].reset_index(drop=True)
        anomaly_types_train = anomaly_types_sorted.iloc[:split_idx].reset_index(drop=True)
        anomaly_types_test = anomaly_types_sorted.iloc[split_idx:].reset_index(drop=True)
        metadata_train = metadata_sorted.iloc[:split_idx].reset_index(drop=True)
        metadata_test = metadata_sorted.iloc[split_idx:].reset_index(drop=True)
        
        print(f"\n📊 Split Statistics:")
        print(f"   Training set: {len(X_train):,} events ({(1-test_size)*100:.0f}%)")
        print(f"   Test set: {len(X_test):,} events ({test_size*100:.0f}%)")
        
        print(f"\n   Training period: {metadata_train['timestamp'].min()} to {metadata_train['timestamp'].max()}")
        print(f"   Test period: {metadata_test['timestamp'].min()} to {metadata_test['timestamp'].max()}")
        
        print(f"\n   Training anomaly rate: {y_train.mean()*100:.2f}%")
        print(f"   Test anomaly rate: {y_test.mean()*100:.2f}%")
        
        # Check anomaly type distribution
        print(f"\n📋 Anomaly Type Distribution:")
        print(f"   Training set:")
        for anom_type, count in anomaly_types_train.value_counts().items():
            if anom_type != 'none':
                pct = (count / len(anomaly_types_train)) * 100
                print(f"      {anom_type:20s}: {count:4d} ({pct:5.2f}%)")
        
        print(f"   Test set:")
        for anom_type, count in anomaly_types_test.value_counts().items():
            if anom_type != 'none':
                pct = (count / len(anomaly_types_test)) * 100
                print(f"      {anom_type:20s}: {count:4d} ({pct:5.2f}%)")
        
        self.preprocessing_stats['train_size'] = len(X_train)
        self.preprocessing_stats['test_size'] = len(X_test)
        self.preprocessing_stats['train_anomaly_rate'] = y_train.mean()
        self.preprocessing_stats['test_anomaly_rate'] = y_test.mean()
        
        return (X_train, X_test, y_train, y_test, 
                anomaly_types_train, anomaly_types_test,
                metadata_train, metadata_test)
    
    def scale_features(self, X_train, X_test):
        """
        Standardize features using StandardScaler.
        
        WHY: Isolation Forest works better with normalized features.
        
        IMPORTANT: Fit scaler ONLY on training data to prevent data leakage.
        Transform both train and test using the SAME scaler (fitted on train).
        
        StandardScaler formula: z = (x - mean) / std
        """
        print("\n" + "="*80)
        print("STEP 5: FEATURE SCALING (STANDARDIZATION)")
        print("="*80)
        
        print(f"\n🔧 Using StandardScaler:")
        print(f"   Formula: z = (x - mean) / std")
        print(f"   Result: Mean ≈ 0, Std ≈ 1 for each feature")
        
        # Fit on training data ONLY
        print(f"\n📐 Fitting scaler on training data...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform test data using training statistics
        print(f"   Transforming test data using training statistics...")
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Verify scaling
        print(f"\n✅ Scaling verification:")
        sample_features = X_train.columns[:3]
        for feat in sample_features:
            train_mean = X_train_scaled[feat].mean()
            train_std = X_train_scaled[feat].std()
            print(f"   {feat:30s}: mean={train_mean:7.4f}, std={train_std:7.4f}")
        
        print(f"\n⚠️  CRITICAL: Scaler fitted on training data only!")
        print(f"   Test data scaled using training mean/std (no leakage)")
        
        return X_train_scaled, X_test_scaled
    
    def save_preprocessed_data(self, X_train, X_test, y_train, y_test,
                               anomaly_types_train, anomaly_types_test,
                               metadata_train, metadata_test, output_dir='data/processed'):
        """
        Save preprocessed data and preprocessing artifacts.
        
        WHY: 
        - Reuse preprocessed data without rerunning pipeline
        - Save scaler for future predictions
        - Maintain reproducibility
        """
        print("\n" + "="*80)
        print("STEP 6: SAVING PREPROCESSED DATA")
        print("="*80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training data
        train_data = X_train.copy()
        train_data['is_anomaly'] = y_train
        train_data['anomaly_type'] = anomaly_types_train
        train_data['timestamp'] = metadata_train['timestamp']
        train_data['session_id'] = metadata_train['session_id']
        train_path = os.path.join(output_dir, 'train_data.csv')
        train_data.to_csv(train_path, index=False)
        print(f"\n✅ Training data saved: {train_path}")
        print(f"   Shape: {X_train.shape}")
        
        # Save test data
        test_data = X_test.copy()
        test_data['is_anomaly'] = y_test
        test_data['anomaly_type'] = anomaly_types_test
        test_data['timestamp'] = metadata_test['timestamp']
        test_data['session_id'] = metadata_test['session_id']
        test_path = os.path.join(output_dir, 'test_data.csv')
        test_data.to_csv(test_path, index=False)
        print(f"✅ Test data saved: {test_path}")
        print(f"   Shape: {X_test.shape}")
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved: {scaler_path}")
        
        # Save label encoders
        encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
        joblib.dump(self.label_encoders, encoders_path)
        print(f"✅ Label encoders saved: {encoders_path}")
        
        # Save feature names
        feature_names_path = os.path.join(output_dir, 'feature_names.txt')
        with open(feature_names_path, 'w') as f:
            f.write('\n'.join(self.feature_names))
        print(f"✅ Feature names saved: {feature_names_path}")
        
        # Save preprocessing statistics
        stats_path = os.path.join(output_dir, 'preprocessing_stats.txt')
        with open(stats_path, 'w') as f:
            f.write("PREPROCESSING STATISTICS\n")
            f.write("="*60 + "\n\n")
            for key, value in self.preprocessing_stats.items():
                f.write(f"{key}: {value}\n")
        print(f"✅ Statistics saved: {stats_path}")
        
        print(f"\n📁 All files saved to: {output_dir}/")
        
        return output_dir
    
    def run_full_pipeline(self, input_filepath, output_dir='data/processed', test_size=0.2):
        """
        Execute complete preprocessing pipeline.
        
        Pipeline Steps:
        1. Load data
        2. Separate features and labels
        3. Encode categorical features
        4. Chronological train/test split
        5. Scale features
        6. Save preprocessed data
        """
        print("\n" + "="*80)
        print("MOBILE APP TELEMETRY - PREPROCESSING PIPELINE")
        print("="*80)
        print(f"\nInput file: {input_filepath}")
        print(f"Output directory: {output_dir}")
        print(f"Test size: {test_size*100:.0f}%")
        print(f"Random state: {self.random_state}")
        
        # Step 1: Load data
        df = self.load_data(input_filepath)
        
        # Step 2: Separate features and labels
        X, y, anomaly_types, metadata = self.separate_features_labels(df)
        
        # Step 3: Encode categorical features
        X = self.encode_categorical_features(X)
        
        # Step 4: Chronological split
        (X_train, X_test, y_train, y_test, 
         anomaly_types_train, anomaly_types_test,
         metadata_train, metadata_test) = self.chronological_split(
            X, y, anomaly_types, metadata, test_size=test_size
        )
        
        # Step 5: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        # Step 6: Save preprocessed data
        output_path = self.save_preprocessed_data(
            X_train_scaled, X_test_scaled, y_train, y_test,
            anomaly_types_train, anomaly_types_test,
            metadata_train, metadata_test, output_dir
        )
        
        # Final summary
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE!")
        print("="*80)
        print(f"\n✅ Training data ready: {len(X_train_scaled):,} events")
        print(f"✅ Test data ready: {len(X_test_scaled):,} events")
        print(f"✅ Features: {X_train_scaled.shape[1]}")
        print(f"✅ Scaler and encoders saved")
        
        print(f"\n📂 Output files:")
        print(f"   {output_dir}/train_data.csv")
        print(f"   {output_dir}/test_data.csv")
        print(f"   {output_dir}/scaler.pkl")
        print(f"   {output_dir}/label_encoders.pkl")
        print(f"   {output_dir}/feature_names.txt")
        print(f"   {output_dir}/preprocessing_stats.txt")
        
        print("\n" + "="*80)
        print("NEXT STEP: Train Isolation Forest model using train_data.csv")
        print("="*80)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'anomaly_types_train': anomaly_types_train,
            'anomaly_types_test': anomaly_types_test,
            'metadata_train': metadata_train,
            'metadata_test': metadata_test,
            'output_dir': output_path
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "data/app_performance_logs_with_labels(v3).csv"
    OUTPUT_DIR = "data/processed"
    TEST_SIZE = 0.2  # 80% train, 20% test
    RANDOM_STATE = 42
    
    # Initialize preprocessor
    preprocessor = TelemetryPreprocessor(random_state=RANDOM_STATE)
    
    # Run full pipeline
    results = preprocessor.run_full_pipeline(
        input_filepath=INPUT_FILE,
        output_dir=OUTPUT_DIR,
        test_size=TEST_SIZE
    )
    
    print("\n" + "="*80)
    print("USAGE EXAMPLE - HOW TO LOAD PREPROCESSED DATA:")
    print("="*80)
    print("""
import pandas as pd
import joblib

# Load preprocessed data
train_data = pd.read_csv('data/processed/train_data.csv')
test_data = pd.read_csv('data/processed/test_data.csv')

# Separate features and labels
feature_cols = [col for col in train_data.columns 
                if col not in ['is_anomaly', 'anomaly_type', 'timestamp', 'session_id']]

X_train = train_data[feature_cols]
y_train = train_data['is_anomaly']

X_test = test_data[feature_cols]
y_test = test_data['is_anomaly']

# Load scaler (if needed for new predictions)
scaler = joblib.load('data/processed/scaler.pkl')

print(f"Training features: {X_train.shape}")
print(f"Test features: {X_test.shape}")
    """)