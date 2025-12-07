# Gravitational Botnet Detector - Usage Examples

This document provides practical examples for using the Gravitational Botnet Detector.

## Quick Start Examples

### 1. Basic Detection

Run detection on a CSV file with default settings:

```bash
python main.py --input data/sample_telemetry.csv
```

### 2. Custom Threshold

Adjust sensitivity by changing the anomaly threshold:

```bash
# More sensitive (catches more anomalies)
python main.py --input data.csv --threshold 0.5

# Less sensitive (fewer false positives)
python main.py --input data.csv --threshold 0.9
```

### 3. Adjust Simulation Parameters

Fine-tune the gravitational simulation:

```bash
# More iterations for better convergence
python main.py --input data.csv --iterations 100

# Adjust gravitational force strength
python main.py --input data.csv -G 1.5

# Combine multiple parameters
python main.py --input data.csv --threshold 0.7 --iterations 75 -G 1.2
```

### 4. Save Results to File

Export detection results:

```bash
python main.py --input data.csv --output results.txt
```

### 5. Quiet Mode

Suppress logging for production use:

```bash
python main.py --input data.csv --quiet
```

### 6. Verbose Mode

Get detailed debugging information:

```bash
python main.py --input data.csv --verbose
```

## Python API Examples

### Basic Detection

```python
from gravitational_botnet_detector import GravitationalDetector

# Initialize with default config
detector = GravitationalDetector()

# Run detection
results = detector.detect_from_csv('telemetry.csv')

# Print report
report = detector.get_anomaly_report(results)
print(report)
```

### Custom Configuration

```python
from gravitational_botnet_detector import GravitationalDetector

config = {
    'anomaly_threshold': 0.75,
    'gravity': {
        'iterations': 100,
        'gravitational_constant': 1.5,
        'damping': 0.85
    }
}

detector = GravitationalDetector(config)
results = detector.detect_from_csv('telemetry.csv')
```

### Access Detailed Results

```python
from gravitational_botnet_detector import GravitationalDetector
import numpy as np

detector = GravitationalDetector()
results = detector.detect_from_csv('telemetry.csv')

# Get anomaly indices
anomaly_indices = np.where(results['anomalies'])[0]
print(f"Anomalous samples: {anomaly_indices}")

# Get top anomaly scores
top_5 = np.argsort(results['anomaly_scores'])[-5:]
print(f"Top 5 anomaly scores: {results['anomaly_scores'][top_5]}")

# Analyze clusters
n_clusters = len(np.unique(results['clusters']))
print(f"Number of clusters: {n_clusters}")
```

### Modular Usage

Use individual components for custom workflows:

```python
from gravitational_botnet_detector import (
    FeatureExtractor,
    GravitationalSimulator
)

# Step 1: Extract features
extractor = FeatureExtractor()
df = extractor.load_csv('telemetry.csv')
features, names = extractor.extract_features(df)

# Step 2: Run simulation
simulator = GravitationalSimulator({'iterations': 50})
sim_results = simulator.simulate(features)

# Step 3: Get clusters
clusters = simulator.compute_cluster_assignments(
    sim_results['final_positions']
)

print(f"Formed {len(np.unique(clusters))} clusters")
```

## Use Cases

### Botnet Detection

Detect coordinated malicious behavior:

```bash
# Use higher iterations for stable clustering
python main.py --input network_logs.csv --iterations 100 --threshold 0.75
```

### APT Detection

Identify Advanced Persistent Threats:

```bash
# Lower threshold to catch subtle anomalies
python main.py --input apt_telemetry.csv --threshold 0.6 --iterations 150
```

### Real-time Monitoring

Process streaming data in batches:

```python
import time
from gravitational_botnet_detector import GravitationalDetector

detector = GravitationalDetector({'anomaly_threshold': 0.7})

while True:
    # Process latest batch
    results = detector.detect_from_csv('latest_batch.csv')
    
    if results['anomalies'].sum() > 0:
        print(f"⚠️  {results['anomalies'].sum()} anomalies detected!")
        # Trigger alerts, logging, etc.
    
    time.sleep(60)  # Check every minute
```

### Batch Analysis

Analyze historical data:

```python
from pathlib import Path
from gravitational_botnet_detector import GravitationalDetector

detector = GravitationalDetector()

# Process all CSV files in a directory
for csv_file in Path('logs/').glob('*.csv'):
    print(f"\nAnalyzing {csv_file.name}")
    results = detector.detect_from_csv(str(csv_file))
    
    n_anomalies = results['anomalies'].sum()
    if n_anomalies > 0:
        print(f"  Found {n_anomalies} anomalies")
        report = detector.get_anomaly_report(results)
        
        # Save report
        output_file = f"reports/{csv_file.stem}_report.txt"
        with open(output_file, 'w') as f:
            f.write(report)
```

### Parameter Tuning

Find optimal parameters for your data:

```python
from gravitational_botnet_detector import GravitationalDetector
import numpy as np

# Test different thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
results_by_threshold = {}

for threshold in thresholds:
    detector = GravitationalDetector({'anomaly_threshold': threshold})
    results = detector.detect_from_csv('validation_data.csv')
    
    n_anomalies = results['anomalies'].sum()
    results_by_threshold[threshold] = n_anomalies
    print(f"Threshold {threshold}: {n_anomalies} anomalies")

# Find optimal threshold
print(f"\nResults: {results_by_threshold}")
```

## Configuration Examples

### High Sensitivity Configuration

For catching subtle anomalies:

```yaml
# config_sensitive.yaml
anomaly_threshold: 0.5
gravity:
  gravitational_constant: 1.2
  iterations: 100
  damping: 0.9
```

```bash
python main.py --input data.csv --config config_sensitive.yaml
```

### Fast Processing Configuration

For quick analysis:

```yaml
# config_fast.yaml
anomaly_threshold: 0.7
gravity:
  gravitational_constant: 1.0
  iterations: 20
  damping: 0.8
```

```bash
python main.py --input data.csv --config config_fast.yaml
```

### Production Configuration

Balanced for accuracy and performance:

```yaml
# config_production.yaml
anomaly_threshold: 0.75
gravity:
  gravitational_constant: 1.0
  iterations: 50
  damping: 0.9
  min_distance: 0.1
```

```bash
python main.py --input data.csv --config config_production.yaml
```

## Troubleshooting

### No Anomalies Detected

- Try lowering the threshold: `--threshold 0.5`
- Increase iterations: `--iterations 100`
- Check that your data has numeric columns

### Too Many False Positives

- Increase the threshold: `--threshold 0.85`
- Adjust gravitational constant: `-G 1.5`
- Use more iterations for better convergence

### Slow Performance

- Reduce iterations: `--iterations 20`
- Process smaller batches of data
- Consider sampling large datasets

## Integration Examples

### SIEM Integration

Export results in JSON format:

```python
import json
from gravitational_botnet_detector import GravitationalDetector

detector = GravitationalDetector()
results = detector.detect_from_csv('telemetry.csv')

# Convert to JSON-friendly format
output = {
    'total_samples': len(results['anomalies']),
    'anomaly_count': int(results['anomalies'].sum()),
    'anomaly_indices': results['anomalies'].nonzero()[0].tolist(),
    'anomaly_scores': results['anomaly_scores'].tolist(),
    'clusters': results['clusters'].tolist()
}

with open('siem_output.json', 'w') as f:
    json.dump(output, f, indent=2)
```

### Alert Integration

Send alerts for high-score anomalies:

```python
from gravitational_botnet_detector import GravitationalDetector
import smtplib

detector = GravitationalDetector({'anomaly_threshold': 0.8})
results = detector.detect_from_csv('telemetry.csv')

high_score_anomalies = results['anomaly_scores'] > 0.9

if high_score_anomalies.sum() > 0:
    # Send alert (example)
    message = f"Critical: {high_score_anomalies.sum()} high-score anomalies detected!"
    # Send via email, Slack, PagerDuty, etc.
    print(message)
```

---

For more information, see the main [README.md](README.md) file.
