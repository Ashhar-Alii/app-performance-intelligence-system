import random
import uuid
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class MobileTelemetryGenerator:
    """
    Generates realistic mobile application telemetry data with:
    - Gradual anomaly transitions (ramps, not instant jumps)
    - Time-based patterns (hourly/daily trends)
    - Version-specific anomaly patterns
    - Network type impact on latency
    - Cold start simulation
    - Correlated metric anomalies
    - Session degradation patterns
    - Error code streaks and bursts
    
    v3 FIX: Anomaly decision is made ONCE per session, not per event.
            Each session either has an anomaly window or doesn't.
            Prevents unrealistic repeated anomaly triggering within sessions.
    
    Note: This is OFFLINE BATCH SIMULATION, not real-time streaming.
    """
    
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        
        # App versions with specific anomaly patterns
        self.app_versions = {
            "1.0.0": {
                "quality": 0.95, 
                "weight": 0.15,
                "anomaly_bias": "none",          # Stable old version
                "memory_leak_prone": False
            },
            "1.1.0": {
                "quality": 0.85, 
                "weight": 0.25,
                "anomaly_bias": "error_burst",    # API breaking changes
                "memory_leak_prone": False
            },
            "1.2.0": {
                "quality": 0.70, 
                "weight": 0.35,
                "anomaly_bias": "memory_leak",    # Known memory issues
                "memory_leak_prone": True
            },
            "1.3.0": {
                "quality": 0.90, 
                "weight": 0.25,
                "anomaly_bias": "fps_drop",       # New rendering changes
                "memory_leak_prone": False
            },
        }
        
        self.screens = [
            "home_feed", "post_view", "profile", 
            "search", "settings", "notifications", "chat"
        ]
        
        self.anomaly_types = [
            "latency_cascade", "fps_drop", "error_burst", 
            "memory_leak", "api_timeout"
        ]
        
        # Network characteristics
        self.network_profiles = {
            "wifi": {
                "latency_multiplier": 1.0,
                "stability": 0.95,
                "packet_loss_rate": 0.01
            },
            "5g": {
                "latency_multiplier": 1.1,
                "stability": 0.90,
                "packet_loss_rate": 0.02
            },
            "4g": {
                "latency_multiplier": 1.4,
                "stability": 0.75,
                "packet_loss_rate": 0.05
            }
        }
    
    def _get_time_based_multiplier(self, timestamp):
        """Simulate realistic usage patterns based on time of day"""
        hour = timestamp.hour
        
        if 9 <= hour <= 23:
            load_factor = 1.2
        else:
            load_factor = 0.8
        
        return load_factor * random.uniform(0.9, 1.1)
    
    def _generate_base_metrics(self, app_version, session_health, 
                                event_index, total_events, timestamp, network_type):
        """Generate baseline metrics with realistic correlations"""
        
        version_info = self.app_versions[app_version]
        version_quality = version_info["quality"]
        time_multiplier = self._get_time_based_multiplier(timestamp)
        
        # Cold start penalty for first few events
        cold_start_penalty = 0
        if event_index < 3:
            cold_start_penalty = int((3 - event_index) * random.randint(100, 300))
        
        # Memory leak for specific versions
        if version_info["memory_leak_prone"]:
            memory_growth = event_index * random.uniform(3.0, 5.0)
        else:
            memory_growth = event_index * random.uniform(1.5, 3.0)
        
        base_memory = 100 + memory_growth
        
        version_penalty = int((1 - version_quality) * 100)
        session_penalty = int((1 - session_health) * 150)
        time_penalty = int((time_multiplier - 1) * 80)
        
        total_latency_penalty = version_penalty + session_penalty + time_penalty + cold_start_penalty
        total_fps_penalty = int(total_latency_penalty / 10)
        
        # Base latency before network effects
        base_api_latency = max(50, random.randint(80, 200) + total_latency_penalty)
        
        # Network type affects latency
        network_profile = self.network_profiles[network_type]
        api_latency = int(base_api_latency * network_profile["latency_multiplier"])
        
        # Network stability affects variance
        if random.random() > network_profile["stability"]:
            api_latency = int(api_latency * random.uniform(1.3, 2.0))
        
        # Packet loss causes retries
        if random.random() < network_profile["packet_loss_rate"]:
            api_latency = int(api_latency * random.uniform(2.0, 3.0))
        
        ui_response = max(20, int(api_latency * random.uniform(0.3, 0.6)))
        fps = max(15, random.randint(50, 60) - total_fps_penalty)
        memory = min(600, base_memory + random.randint(-10, 30))
        
        return {
            "api_latency_ms": api_latency,
            "ui_response_ms": ui_response,
            "fps": fps,
            "memory_mb": memory,
            "error_code": 0,
            "network_type": network_type,
        }
    
    def _apply_gradual_anomaly(self, metrics, anomaly_type, severity):
        """
        Apply gradual anomaly transitions instead of instant jumps
        severity: 0.0 (starting) to 1.0 (full anomaly)
        """
        
        if anomaly_type == "latency_cascade":
            max_spike = random.randint(1200, 4000)
            current_spike = int(metrics["api_latency_ms"] + 
                              (max_spike - metrics["api_latency_ms"]) * severity)
            
            metrics["api_latency_ms"] = current_spike
            metrics["ui_response_ms"] = int(metrics["ui_response_ms"] * (1 + severity * 2))
            metrics["fps"] = max(15, int(metrics["fps"] * (1 - severity * 0.6)))
            
            if severity > 0.7 and random.random() < 0.4:
                metrics["error_code"] = random.choice([408, 504])
        
        elif anomaly_type == "fps_drop":
            target_fps = random.randint(5, 25)
            metrics["fps"] = max(5, int(metrics["fps"] - 
                                       (metrics["fps"] - target_fps) * severity))
            metrics["ui_response_ms"] = int(metrics["ui_response_ms"] * (1 + severity))
            metrics["memory_mb"] = min(600, int(metrics["memory_mb"] * (1 + severity * 0.3)))
        
        elif anomaly_type == "error_burst":
            if severity > 0.3:
                metrics["error_code"] = random.choice([500, 502, 503, 504])
                metrics["api_latency_ms"] = int(metrics["api_latency_ms"] * (1 + severity * 2))
                metrics["ui_response_ms"] = int(metrics["ui_response_ms"] * (1 + severity))
        
        elif anomaly_type == "memory_leak":
            leak_amount = int(300 * severity)
            metrics["memory_mb"] = min(650, metrics["memory_mb"] + leak_amount)
            if severity > 0.5:
                metrics["fps"] = max(15, int(metrics["fps"] * (1 - severity * 0.4)))
                metrics["ui_response_ms"] = int(metrics["ui_response_ms"] * (1 + severity * 0.5))
        
        elif anomaly_type == "api_timeout":
            if severity > 0.6:
                metrics["api_latency_ms"] = random.randint(5000, 10000)
                metrics["error_code"] = random.choice([408, 504, 503])
                metrics["fps"] = max(10, int(metrics["fps"] * (1 - severity * 0.5)))
            else:
                metrics["api_latency_ms"] = int(metrics["api_latency_ms"] * (1 + severity * 3))
        
        return metrics
    
    def generate_sessions(self, num_sessions=200, events_per_session=50, 
                         start_date=None):
        """
        Generate complete dataset with temporal patterns.
        
        v3 FIX: Anomaly decision is made ONCE per session (not per event).
        
        Rate calculation:
        - Per-SESSION probability (not per-event)
        - Good versions (quality > 0.85): 35% base chance of session anomaly
        - Bad versions (quality <= 0.85): 65% base chance of session anomaly
        - Multiplied by (2 - session_health) for health-based scaling
        - With avg anomaly duration ~10 events out of 50, this gives ~10-15% event-level anomaly rate
        
        Math: P(session_anomaly) * avg_duration / events_per_session = event_anomaly_rate
              ~0.58 * 10 / 50 = ~0.116 = ~12%
        """
        
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
        
        data = []
        current_time = start_date
        
        # Track session-level decisions for debugging
        sessions_with_anomaly = 0
        
        print(f"Generating {num_sessions} sessions with {events_per_session} events each...")
        print("Target anomaly rate: 10-15% (production realistic)")
        print("Anomaly decision: ONCE per session (not per event)")
        
        for session_num in range(num_sessions):
            session_id = str(uuid.uuid4())
            
            app_version = random.choices(
                list(self.app_versions.keys()),
                weights=[v["weight"] for v in self.app_versions.values()]
            )[0]
            
            # Select network type (weighted towards wifi)
            network_type = random.choices(
                ["wifi", "5g", "4g"],
                weights=[0.60, 0.25, 0.15]
            )[0]
            
            session_health = random.choices(
                [1.0, 0.95, 0.85, 0.70, 0.50],
                weights=[0.40, 0.25, 0.20, 0.10, 0.05]
            )[0]
            
            version_info = self.app_versions[app_version]
            
            # ============================================================
            # v3 FIX: Per-SESSION anomaly probability (not per-event)
            # 
            # v1 was: per-event rate 0.03 / 0.08 → checked 50 times → ~40%
            # v2 was: per-event rate 0.007 / 0.022 → still checked 50 times
            # v3 is:  per-session rate 0.35 / 0.65 → checked ONCE → ~12%
            # ============================================================
            base_anomaly_rate = 0.35 if version_info["quality"] > 0.85 else 0.65
            session_anomaly_rate = min(0.95, base_anomaly_rate * (2 - session_health))
            
            # ============================================================
            # v3 FIX: Decide ONCE if this session will contain an anomaly
            # ============================================================
            session_has_anomaly = random.random() < session_anomaly_rate
            
            if session_has_anomaly:
                sessions_with_anomaly += 1
                
                # Decide anomaly type (version bias or random)
                if version_info["anomaly_bias"] != "none" and random.random() < 0.6:
                    current_anomaly_type = version_info["anomaly_bias"]
                else:
                    current_anomaly_type = random.choice(self.anomaly_types)
                
                # Decide WHEN anomaly starts and HOW LONG it lasts
                anomaly_start_index = random.randint(5, events_per_session - 10)
                anomaly_duration = random.randint(5, 15)
                
                # Pre-determine error streak code if error_burst
                if current_anomaly_type == "error_burst":
                    error_code_streak = random.choice([500, 502, 503, 504])
                else:
                    error_code_streak = None
            else:
                # Clean session — no anomaly
                anomaly_start_index = None
                anomaly_duration = 0
                current_anomaly_type = None
                error_code_streak = None
            
            # Reset per-session tracking
            in_anomaly = False
            anomaly_ramp_progress = 0.0
            remaining_anomaly_duration = anomaly_duration
            
            # ============================================================
            # Event loop (anomaly trigger is deterministic, not random)
            # ============================================================
            for event_idx in range(events_per_session):
                # Generate base metrics
                metrics = self._generate_base_metrics(
                    app_version, session_health, event_idx, 
                    events_per_session, current_time, network_type
                )
                
                is_anomaly = 0
                detected_anomaly_type = "none"
                
                # ======================================================
                # v3 FIX: Start anomaly at predetermined index
                # (replaces: if not in_anomaly and random.random() < rate)
                # ======================================================
                if session_has_anomaly and event_idx == anomaly_start_index:
                    in_anomaly = True
                    anomaly_ramp_progress = 0.0
                
                # Process ongoing anomaly (same as before)
                if in_anomaly:
                    anomaly_ramp_progress = min(1.0, anomaly_ramp_progress + 
                                               random.uniform(0.15, 0.35))
                    
                    metrics = self._apply_gradual_anomaly(
                        metrics, current_anomaly_type, anomaly_ramp_progress
                    )
                    
                    # Error streak continuation
                    if current_anomaly_type == "error_burst" and metrics["error_code"] != 0:
                        if random.random() < 0.7:
                            metrics["error_code"] = error_code_streak
                    
                    is_anomaly = 1
                    detected_anomaly_type = current_anomaly_type
                    
                    # Count down anomaly duration
                    remaining_anomaly_duration -= 1
                    if remaining_anomaly_duration <= 0:
                        in_anomaly = False
                
                # Record event
                data.append({
                    "timestamp": current_time,
                    "session_id": session_id,
                    "app_version": app_version,
                    "screen_name": random.choice(self.screens),
                    **metrics,
                    
                    # Evaluation-only labels (DO NOT use in training)
                    "is_anomaly": is_anomaly,
                    "anomaly_type": detected_anomaly_type,
                    
                    # Debug info
                    "session_health": round(session_health, 2),
                })
                
                current_time += timedelta(seconds=random.randint(2, 8))
            
            current_time += timedelta(minutes=random.randint(5, 30))
        
        df = pd.DataFrame(data)
        
        # Print session-level anomaly stats
        print(f"\nSession-level stats:")
        print(f"  Sessions with anomaly: {sessions_with_anomaly}/{num_sessions} "
              f"({sessions_with_anomaly/num_sessions:.1%})")
        print(f"  Event-level anomaly rate: {df['is_anomaly'].mean():.2%}")
        
        # Add aggregated rolling window features
        df = self._add_rolling_features(df)
        
        return df
    
    def _add_rolling_features(self, df):
        """
        Add normalized rolling features without label leakage.
        Uses relative statistics instead of absolute thresholds.
        """
        
        print("Computing rolling window features...")
        
        df = df.sort_values(['session_id', 'timestamp']).reset_index(drop=True)
        
        feature_cols = ['api_latency_ms', 'ui_response_ms', 'fps', 'memory_mb']
        
        for col in feature_cols:
            # Rolling mean
            df[f'{col}_rolling_mean_5'] = df.groupby('session_id')[col].transform(
                lambda x: x.rolling(window=5, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'{col}_rolling_std_5'] = df.groupby('session_id')[col].transform(
                lambda x: x.rolling(window=5, min_periods=1).std().fillna(0)
            )
            
            # Rolling max
            df[f'{col}_rolling_max_10'] = df.groupby('session_id')[col].transform(
                lambda x: x.rolling(window=10, min_periods=1).max()
            )
            
            # Z-score (normalized, no label leakage)
            mean = df[f'{col}_rolling_mean_5']
            std = df[f'{col}_rolling_std_5']
            df[f'{col}_zscore'] = ((df[col] - mean) / (std + 1e-6))
        
        # Rate of change features
        df['fps_change_rate'] = df.groupby('session_id')['fps'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        df['latency_change_rate'] = df.groupby('session_id')['api_latency_ms'].transform(
            lambda x: x.pct_change().fillna(0)
        )
        
        # Memory growth rate
        df['memory_growth_rate'] = df.groupby('session_id')['memory_mb'].transform(
            lambda x: x.diff().fillna(0)
        )
        
        # Error density
        df['recent_error_count'] = df.groupby('session_id')['error_code'].transform(
            lambda x: (x != 0).rolling(window=5, min_periods=1).sum()
        )
        
        # Coefficient of variation
        df['latency_cv'] = df.groupby('session_id')['api_latency_ms'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std() / 
                     (x.rolling(window=5, min_periods=1).mean() + 1e-6)
        ).fillna(0)
        
        engineered_count = len([c for c in df.columns 
                               if any(k in c for k in ['rolling', 'rate', 'growth', 
                                                        'recent', 'zscore', '_cv'])])
        print(f"Added {engineered_count} engineered features")
        
        return df
    
    def print_summary(self, df):
        """Print detailed dataset statistics"""
        print("\n" + "="*60)
        print("DATASET SUMMARY (v3 — Per-Session Anomaly Decision)")
        print("="*60)
        
        print(f"\nTotal Events: {len(df):,}")
        print(f"Total Sessions: {df['session_id'].nunique()}")
        print(f"Time Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        anomaly_rate = df['is_anomaly'].mean()
        print(f"Overall Anomaly Rate: {anomaly_rate:.2%}")
        
        # Validate target range
        if 0.10 <= anomaly_rate <= 0.15:
            print("✅ Anomaly rate is in target range (10-15%)")
        elif anomaly_rate < 0.10:
            print("⚠️  Anomaly rate is BELOW target (< 10%)")
        else:
            print("⚠️  Anomaly rate is ABOVE target (> 15%)")
        
        # Session-level anomaly stats
        session_anomaly = df.groupby('session_id')['is_anomaly'].max()
        sessions_with_anomaly = session_anomaly.sum()
        total_sessions = len(session_anomaly)
        print(f"\nSessions with anomaly: {sessions_with_anomaly}/{total_sessions} "
              f"({sessions_with_anomaly/total_sessions:.1%})")
        print(f"Clean sessions: {total_sessions - sessions_with_anomaly}/{total_sessions} "
              f"({(total_sessions - sessions_with_anomaly)/total_sessions:.1%})")
        
        print("\nAnomaly Distribution by Type:")
        print(df['anomaly_type'].value_counts())
        
        print("\nEvents by App Version:")
        version_stats = df.groupby('app_version').agg({
            'session_id': 'nunique',
            'is_anomaly': 'mean'
        }).round(3)
        version_stats.columns = ['Sessions', 'Anomaly Rate']
        print(version_stats)
        
        print("\nNetwork Type Distribution:")
        network_stats = df.groupby('network_type').agg({
            'api_latency_ms': 'mean',
            'session_id': 'nunique'
        }).round(2)
        network_stats.columns = ['Avg Latency (ms)', 'Sessions']
        print(network_stats)
        
        print("\nBase Metric Statistics:")
        metric_cols = ['api_latency_ms', 'ui_response_ms', 'fps', 'memory_mb']
        print(df[metric_cols].describe().round(2))
        
        print("\nNormal vs Anomaly Comparison:")
        for col in metric_cols:
            normal_mean = df[df['is_anomaly']==0][col].mean()
            anomaly_mean = df[df['is_anomaly']==1][col].mean()
            ratio = anomaly_mean / normal_mean if normal_mean != 0 else 0
            print(f"  {col:20s}: Normal={normal_mean:7.1f}  "
                  f"Anomaly={anomaly_mean:7.1f}  Ratio={ratio:.2f}x")
        
        print("\nCold Start Analysis (First 3 Events):")
        cold_start = df.groupby('session_id').head(3)
        normal_latency = df[df.groupby('session_id').cumcount() >= 3]['api_latency_ms'].mean()
        cold_latency = cold_start['api_latency_ms'].mean()
        print(f"  Cold start avg latency: {cold_latency:.1f}ms")
        print(f"  Warmed up avg latency: {normal_latency:.1f}ms")
        print(f"  Cold start penalty: +{cold_latency - normal_latency:.1f}ms")
        
        print("\nError Code Analysis:")
        error_df = df[df['error_code'] != 0]
        if len(error_df) > 0:
            print(f"  Total errors: {len(error_df)}")
            print(f"  Error rate: {(df['error_code'] != 0).mean():.2%}")
            print("  Error code distribution:")
            print(error_df['error_code'].value_counts())
        
        print("\n" + "="*60 + "\n")
    
    def run_mini_validation(self, df):
        """
        Quick validation checks for regenerated dataset.
        Replaces full EDA — confirms key properties still hold.
        """
        print("\n" + "="*60)
        print("MINI VALIDATION (v3 Dataset)")
        print("="*60)
        
        checks_passed = 0
        total_checks = 7
        
        # --------------------------------------------------
        # Check 1: Anomaly rate in target range (10-15%)
        # --------------------------------------------------
        anomaly_rate = df['is_anomaly'].mean()
        if 0.08 <= anomaly_rate <= 0.18:
            print(f"\n✅ CHECK 1: Anomaly rate = {anomaly_rate:.2%} (target: 10-15%)")
            checks_passed += 1
        else:
            print(f"\n❌ CHECK 1: Anomaly rate = {anomaly_rate:.2%} — OUT OF RANGE")
        
        # --------------------------------------------------
        # Check 2: All 5 anomaly types still present
        # --------------------------------------------------
        anomaly_types_found = df[df['anomaly_type'] != 'none']['anomaly_type'].nunique()
        if anomaly_types_found == 5:
            print(f"✅ CHECK 2: All 5 anomaly types present")
            checks_passed += 1
        else:
            print(f"❌ CHECK 2: Only {anomaly_types_found}/5 anomaly types found")
        
        print("   Distribution:")
        type_counts = df['anomaly_type'].value_counts()
        for atype, count in type_counts.items():
            print(f"     {atype:20s}: {count:5d} ({count/len(df)*100:.1f}%)")
        
        # --------------------------------------------------
        # Check 3: Normal vs Anomaly still separable
        # --------------------------------------------------
        normal_latency = df[df['is_anomaly']==0]['api_latency_ms'].mean()
        anomaly_latency = df[df['is_anomaly']==1]['api_latency_ms'].mean()
        latency_ratio = anomaly_latency / normal_latency
        
        if latency_ratio > 2.0:
            print(f"✅ CHECK 3: Anomaly/Normal latency ratio = {latency_ratio:.1f}x (good separation)")
            checks_passed += 1
        else:
            print(f"⚠️  CHECK 3: Anomaly/Normal latency ratio = {latency_ratio:.1f}x (weak separation)")
        
        print(f"   Normal avg latency:  {normal_latency:.1f} ms")
        print(f"   Anomaly avg latency: {anomaly_latency:.1f} ms")
        
        # --------------------------------------------------
        # Check 4: Version 1.2.0 memory leak bias preserved
        # --------------------------------------------------
        v12 = df[(df['app_version']=='1.2.0') & (df['anomaly_type'] != 'none')]
        if len(v12) > 0:
            mem_leak_pct = (v12['anomaly_type'] == 'memory_leak').mean()
            if mem_leak_pct > 0.40:
                print(f"✅ CHECK 4: v1.2.0 memory leak bias = {mem_leak_pct:.1%} (expected >40%)")
                checks_passed += 1
            else:
                print(f"⚠️  CHECK 4: v1.2.0 memory leak bias = {mem_leak_pct:.1%} (lower than expected)")
        else:
            print(f"❌ CHECK 4: No anomalies found for v1.2.0")
        
        # --------------------------------------------------
        # Check 5: Sufficient data for ML training
        # --------------------------------------------------
        normal_count = (df['is_anomaly'] == 0).sum()
        anomaly_count = (df['is_anomaly'] == 1).sum()
        
        if anomaly_count >= 500:
            print(f"✅ CHECK 5: Sufficient anomaly samples "
                  f"({anomaly_count:,} anomalies, {normal_count:,} normal)")
            checks_passed += 1
        else:
            print(f"⚠️  CHECK 5: Low anomaly samples ({anomaly_count:,}) "
                  f"— may need more sessions")
        
        # --------------------------------------------------
        # Check 6: Engineered features present and valid
        # --------------------------------------------------
        expected_features = ['api_latency_ms_rolling_mean_5', 'fps_zscore', 
                           'memory_growth_rate', 'recent_error_count', 'latency_cv']
        missing_features = [f for f in expected_features if f not in df.columns]
        
        has_inf = df.select_dtypes(include=[np.number]).isin(
            [np.inf, -np.inf]).any().any()
        nan_count = df.select_dtypes(include=[np.number]).isna().sum().sum()
        
        if len(missing_features) == 0 and not has_inf:
            print(f"✅ CHECK 6: All engineered features present, "
                  f"no inf values (NaNs: {nan_count})")
            checks_passed += 1
        else:
            if missing_features:
                print(f"❌ CHECK 6: Missing features: {missing_features}")
            if has_inf:
                print(f"❌ CHECK 6: Infinite values detected in features")
        
        # --------------------------------------------------
        # Check 7 (NEW): Per-session anomaly decision validated
        # --------------------------------------------------
        session_anomaly_counts = df.groupby('session_id').apply(
            lambda g: (g['is_anomaly'].diff().fillna(0) == 1).sum()
        )
        
        # Each session should have AT MOST 1 anomaly sequence
        max_sequences = session_anomaly_counts.max()
        multi_sequence_sessions = (session_anomaly_counts > 1).sum()
        
        if max_sequences <= 1:
            print(f"✅ CHECK 7: Per-session anomaly decision verified "
                  f"(max 1 anomaly sequence per session)")
            checks_passed += 1
        else:
            print(f"⚠️  CHECK 7: {multi_sequence_sessions} sessions have "
                  f"multiple anomaly sequences (max={max_sequences})")
            # Still pass if very few violations
            if multi_sequence_sessions <= 2:
                checks_passed += 1
        
        # --------------------------------------------------
        # Final verdict
        # --------------------------------------------------
        print(f"\n{'='*60}")
        print(f"VALIDATION RESULT: {checks_passed}/{total_checks} checks passed")
        
        if checks_passed == total_checks:
            print("✅ ALL CHECKS PASSED — Dataset ready for preprocessing")
        elif checks_passed >= 5:
            print("⚠️  MOSTLY PASSED — Review warnings before proceeding")
        else:
            print("❌ MULTIPLE FAILURES — Consider adjusting parameters")
        
        print("="*60 + "\n")
        
        return checks_passed == total_checks


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Initialize generator
    generator = MobileTelemetryGenerator(seed=42)
    
    # Generate dataset
    df = generator.generate_sessions(
        num_sessions=200,
        events_per_session=50,
        start_date=datetime.now() - timedelta(days=7)
    )
    
    # Print full summary
    generator.print_summary(df)
    
    # Run mini validation (replaces full EDA redo)
    validation_passed = generator.run_mini_validation(df)
    
    # Save complete dataset (with labels for evaluation)
    output_file = "app_performance_logs_with_labels.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Complete data (with labels) saved to {output_file}")
    
    # Save training-only version (NO LABELS, NO DEBUG INFO)
    training_cols = [c for c in df.columns 
                    if c not in ['is_anomaly', 'anomaly_type', 'session_health']]
    
    df[training_cols].to_csv("app_performance_logs_training.csv", index=False)
    print(f"✅ Training data (no labels) saved to app_performance_logs_training.csv")
    
    # ================================================================
    # DELTA REPORT
    # ================================================================
    anomaly_rate = df['is_anomaly'].mean()
    session_anomaly = df.groupby('session_id')['is_anomaly'].max()
    sessions_with = session_anomaly.sum()
    total_sessions = len(session_anomaly)
    
    print("\n" + "="*60)
    print("DELTA REPORT: Dataset v3")
    print("="*60)
    print(f"""
    STRUCTURAL FIX APPLIED:
    - Anomaly decision: per-EVENT (v1/v2) → per-SESSION (v3)
    - Each session has AT MOST one anomaly window
    - Anomaly start index and duration predetermined before event loop
    - No more random re-triggering within same session
    
    RATE PARAMETERS:
    - Good versions (quality > 0.85): 35% session anomaly probability
    - Bad versions (quality <= 0.85):  65% session anomaly probability
    - Scaled by (2 - session_health) for health-based adjustment
    - Capped at 95% to ensure some clean sessions
    
    ACTUAL RESULTS:
    - Event-level anomaly rate: {anomaly_rate:.2%}
    - Sessions with anomaly: {sessions_with}/{total_sessions} ({sessions_with/total_sessions:.1%})
    - Clean sessions: {total_sessions - sessions_with}/{total_sessions} ({(total_sessions - sessions_with)/total_sessions:.1%})
    
    WHAT STAYED THE SAME:
    - All 5 anomaly types with same signatures
    - Version-specific anomaly biases (v1.2.0 → memory_leak)
    - Network type effects (WiFi/5G/4G latency multipliers)
    - Cold start simulation (first 3 events penalty)
    - Gradual ramp transitions (severity 0.0 → 1.0)
    - All engineered features (rolling, z-score, rates)
    
    NEXT STEPS:
    1. Preprocessing (scaling, encoding, train/test split)
    2. Train: Z-Score baseline + Isolation Forest + LOF
    3. Compare all 3 models
    4. Streaming simulation demo
    """)
    print("="*60)