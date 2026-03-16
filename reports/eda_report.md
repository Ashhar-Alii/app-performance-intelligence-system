# 📊 Exploratory Data Analysis (EDA) Report

---

## 1. Dataset Overview

| Metric | Value |
|---|---|
| **Total Events** | 10,000 |
| **Total Sessions** | 200 |
| **Total Features** | 34 |
| **Date Range** | Feb 19 – Feb 22, 2026 |
| **Row-level Anomaly Rate** | 11.91% |
| **Sessions Containing Anomaly** | 59% |

- The dataset represents **simulated mobile application telemetry events** with performance metrics collected across multiple user sessions.
- Each session contains **~50 events**.

---

## 2. Data Quality Checks

### ✅ Missing Values

| Check | Result |
|---|---|
| Missing values | **0** |
| Missing percentage | **0%** |

> All features are complete.

### ✅ Duplicate Rows

| Check | Result |
|---|---|
| Duplicate rows | **0** |

> No duplicated records were detected.

### ✅ Chronological Ordering

- All events within each session follow proper timestamp order.
- Sessions with ordering issues: **0**

---

## 3. Feature Range Validation

| Feature | Min | Max | Expected Range | Valid |
|---|---|---|---|---|
| `api_latency_ms` | 67 | 9,950 | 0 – 10,000 | ✔ |
| `ui_response_ms` | 21 | 1,135 | 0 – 2,000 | ✔ |
| `fps` | 5 | 61 | 0 – 60 | ⚠ Slight overflow |
| `memory_mb` | 90 | 650 | 0 – 650 | ✔ |

> **Note:** The FPS column contains a few values slightly above 60 due to simulation noise.

---

## 4. Distribution Analysis

### 📈 API Latency

| Statistic | Value |
|---|---|
| Mean | 355 ms |
| Median | 200 ms |
| Skewness | 8.07 |
| Kurtosis | 70.45 |

- API latency is **heavily right-skewed**, indicating rare extreme latency spikes.

### 🎮 FPS Distribution

| Statistic | Value |
|---|---|
| Mean | 48 |
| Median | 51 |
| Skewness | -1.96 |

- FPS distribution is **left-skewed**, caused by occasional frame drops.

### 💾 Memory Usage

| Statistic | Value |
|---|---|
| Mean | 192 MB |
| Median | 175 MB |
| Skewness | 2.47 |

- Memory usage shows a **positive skew** due to memory leak anomalies.

---

## 5. Outlier Detection (IQR Method)

| Feature | Outliers | Percentage |
|---|---|---|
| `api_latency_ms` | 1,252 | 12.52% |
| `ui_response_ms` | 863 | 8.63% |
| `fps` | 1,244 | 12.44% |
| `memory_mb` | 480 | 4.80% |

> Outliers are **expected** because anomaly sequences intentionally produce extreme values.

---

## 6. Correlation Analysis

| Feature Pair | Correlation |
|---|---|
| API latency vs UI response | **0.19** |
| API latency vs FPS | **-0.37** |
| UI response vs FPS | **-0.59** |
| Memory vs FPS | **-0.17** |

**Interpretation:**

- Higher latency → **lower FPS**
- UI delay **strongly affects** frame rendering

---

## 7. Normal vs Anomaly Behavior

| Metric | Normal | Anomaly | Change |
|---|---|---|---|
| API latency | 237 ms | 1,228 ms | **+417%** |
| UI response | 106 ms | 155 ms | **+46%** |
| FPS | 49.8 | 36.0 | **-27%** |
| Memory | 178 MB | 296 MB | **+66%** |

> Anomalies **clearly impact** all performance metrics.

---

## 8. Anomaly Sequence Analysis

| Metric | Value |
|---|---|
| Total anomaly sequences | 118 |
| Average sequence length | 10 events |
| Min length | 5 |
| Max length | 15 |

- Anomalies occur as **short bursts** rather than isolated events, which reflects real production incidents.

---

## 9. App Version Analysis

| Version | Avg API Latency |
|---|---|
| 1.2.0 | 406 ms |
| 1.1.0 | 353 ms |
| 1.0.0 | 349 ms |
| **1.3.0** | **290 ms (best)** |

> ⚠ Version **1.2.0** shows the **worst performance**.

---

## 10. Memory Leak Investigation

| Version | Memory Leak Share |
|---|---|
| **1.2.0** | **71.3%** |
| 1.0.0 | 42.6% |
| 1.1.0 | 10.2% |
| 1.3.0 | 7.9% |

> This confirms that version **1.2.0 contains a memory leak pattern**.

---

## 11. Network Performance

| Network | Avg Latency |
|---|---|
| WiFi | 340 ms |
| 5G | 361 ms |
| 4G | 428 ms |

- 4G shows **25.7% higher latency** than WiFi.

---

## 12. Session Health Impact

| Session Health | Anomaly Rate |
|---|---|
| 1.00 | 11.1% |
| 0.95 | 10.7% |
| 0.85 | 14.6% |
| 0.70 | 10.2% |
| **0.50** | **22%** |

- Lower health sessions experience **significantly more anomalies**.

---

## 13. Cold Start Effect

| Event Index | Avg Latency |
|---|---|
| Event 0 | **900 ms** |
| Event 5 | 204 ms |

- Cold start latency is **339% higher** than steady-state performance.

---

## 14. Error Code Analysis

| Metric | Value |
|---|---|
| Total error events | 467 |
| Error streak rate | 40.9% |
| Avg streak length | 1.69 |

- Errors frequently appear in **short bursts**.

---

## 15. Latency vs Errors

| Condition | Avg Latency |
|---|---|
| **With errors** | **2,496 ms** |
| Without errors | 250 ms |

- High latency **strongly correlates** with backend failures.
- Correlation coefficient: **0.53**

---

## 16. Final Dataset Statistics

| Metric | Value |
|---|---|
| Total events | 10,000 |
| Anomalies | 1,191 |
| Row anomaly rate | 11.91% |
| Sessions with anomaly | 118 / 200 (59%) |

---

## 🔑 Key Findings

- 📌 API latency spikes increase by **417%** during anomalies.
- 📌 FPS drops by **27%** during performance issues.
- 📌 Version **1.2.0** introduces a **memory leak pattern**.
- 📌 **4G networks** show higher latency than WiFi.
- 📌 Cold starts introduce **~700 ms** latency overhead.
- 📌 Error events **strongly correlate** with extreme API latency.

---

## ✅ Conclusion

The dataset contains **realistic performance degradation patterns** including:

- ⚡ Latency spikes
- 🎮 FPS drops
- 💾 Memory leaks
- 🌐 API timeouts

> Clear statistical differences between normal and anomalous events confirm that the dataset is **suitable for training an anomaly detection model**.