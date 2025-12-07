# Gravitational Botnet Detector

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.7+-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

**Real-time anomaly detection using N-Body gravitational clustering for botnets and APTs.**

The Gravitational Botnet Detector is a lightweight Python package that applies physics-inspired N-Body gravitational simulation to detect anomalous network behavior patterns. By treating network entities as gravitational bodies, the system identifies outliers that don't cluster with normal traffic patterns.

## 🌟 Features

- **N-Body Physics Simulation**: Gravitational clustering treats network entities as masses in space
- **Automated Feature Extraction**: Extracts behavioral features from CSV telemetry data
- **Anomaly Scoring**: Multi-factor scoring based on gravitational binding and cluster isolation
- **CLI Interface**: Easy-to-use command-line tool
- **Configurable**: YAML-based configuration for all parameters
- **Lightweight**: Minimal dependencies (numpy, pandas, PyYAML)
- **Demo-Ready**: Includes sample data and usage examples

## 🔬 How It Works

### The Gravitational Model

1. **Feature Extraction**: Network telemetry data is converted into feature vectors
2. **Gravitational Initialization**: Each entity becomes a "body" with mass and position
3. **N-Body Simulation**: Bodies interact via gravitational forces over multiple iterations
4. **Cluster Formation**: Similar entities gravitationally attract and form clusters
5. **Anomaly Detection**: Isolated entities with weak gravitational binding are flagged as anomalies

**Key Insight**: Normal network behavior clusters together under gravitational attraction, while anomalous behavior (botnets, APTs) remains isolated or forms small, weakly-bound groups.

### Anomaly Scoring Factors

- **Isolation**: Distance from cluster center and nearest neighbors
- **Cluster Size**: Membership in unusually small groups
- **Gravitational Binding**: Weak gravitational ties to cluster
- **Stability**: High kinetic energy indicating unstable behavior

## 📦 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Package Structure

```
gravitational-botnet-detector/
├── gravitational_botnet_detector/
│   ├── __init__.py          # Package initialization
│   ├── features.py          # Feature extraction module
│   ├── gravity.py           # Gravitational simulation
│   └── detector.py          # Main detector class
├── main.py                  # CLI entry point
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
├── data/                    # Sample data (optional)
└── README.md               # This file
```

## 🚀 Quick Start

### Basic Usage

```bash
python main.py --input telemetry.csv
```

### With Configuration

```bash
python main.py --input telemetry.csv --config config.yaml
```

### Custom Parameters

```bash
python main.py --input data.csv --threshold 0.8 --iterations 100
```

### Save Results to File

```bash
python main.py --input data.csv --output results.txt
```

## 📊 Input Data Format

The detector accepts CSV files with network telemetry data. The system is flexible and works with various formats:

### Example CSV Format

```csv
timestamp,src_ip,dst_ip,bytes_sent,bytes_received,packets,duration
1234567890,192.168.1.10,10.0.0.5,1024,2048,15,2.5
1234567891,192.168.1.11,10.0.0.6,512,1024,8,1.2
1234567892,192.168.1.10,10.0.0.5,2048,4096,25,3.1
```

### Supported Columns (Flexible)

The detector automatically extracts features from numeric columns. Common fields include:

- **Temporal**: timestamp, duration, time_delta
- **Network**: src_ip, dst_ip, src_port, dst_port, protocol
- **Volume**: bytes_sent, bytes_received, packets, connections
- **Behavioral**: connection_rate, unique_destinations, failed_attempts

## ⚙️ Configuration

Edit `config.yaml` to customize detection parameters:

```yaml
# Anomaly detection threshold (0-1)
anomaly_threshold: 0.7

# Gravitational simulation parameters
gravity:
  gravitational_constant: 1.0
  iterations: 50
  time_step: 0.1
  damping: 0.9
  min_distance: 0.1
```

### Key Parameters

- **anomaly_threshold**: Score above which samples are flagged (0-1)
  - Higher = stricter (fewer false positives)
  - Lower = more sensitive (catch more anomalies)
  
- **gravitational_constant**: Controls force strength in simulation
  - Higher = stronger clustering
  - Lower = weaker interactions
  
- **iterations**: Number of simulation steps
  - More iterations = better convergence
  - Recommended: 50-100

- **damping**: Velocity damping factor (0-1)
  - Higher = less energy loss
  - Lower = faster stabilization

## 🎯 Use Cases

### 1. Botnet Detection

Identify coordinated malicious actors exhibiting synchronized behavior patterns.

```bash
python main.py --input botnet_traffic.csv --threshold 0.75
```

### 2. APT Detection

Detect Advanced Persistent Threats with unusual communication patterns.

```bash
python main.py --input network_logs.csv --iterations 100
```

### 3. Network Anomaly Monitoring

Real-time monitoring for any unusual network activity.

```bash
python main.py --input realtime_telemetry.csv --config production.yaml
```

## 📈 Example Output

```
============================================================
GRAVITATIONAL BOTNET DETECTOR - ANOMALY REPORT
============================================================
Total Samples: 1000
Anomalies Detected: 15 (1.5%)
Number of Clusters: 3
------------------------------------------------------------
ANOMALOUS SAMPLES:
Index      Score      Cluster   
------------------------------------------------------------
42         0.8523     5         
127        0.8192     5         
234        0.7845     8         
...
============================================================
```

## 🔧 Python API Usage

You can also use the detector programmatically:

```python
from gravitational_botnet_detector import GravitationalDetector

# Initialize detector
config = {
    'anomaly_threshold': 0.7,
    'gravity': {
        'iterations': 50,
        'gravitational_constant': 1.0
    }
}
detector = GravitationalDetector(config)

# Run detection
results = detector.detect_from_csv('telemetry.csv')

# Access results
print(f"Anomalies: {results['anomalies'].sum()}")
print(f"Scores: {results['anomaly_scores']}")
print(f"Clusters: {results['clusters']}")

# Generate report
report = detector.get_anomaly_report(results)
print(report)
```

## 🧪 Testing

Create sample telemetry data for testing:

```python
import pandas as pd
import numpy as np

# Generate normal traffic
normal = pd.DataFrame({
    'bytes_sent': np.random.normal(1000, 200, 100),
    'bytes_received': np.random.normal(2000, 400, 100),
    'packets': np.random.poisson(10, 100),
    'duration': np.random.exponential(2.0, 100)
})

# Add some anomalies
anomalies = pd.DataFrame({
    'bytes_sent': np.random.normal(5000, 100, 10),
    'bytes_received': np.random.normal(10000, 200, 10),
    'packets': np.random.poisson(50, 10),
    'duration': np.random.exponential(0.5, 10)
})

# Combine and save
data = pd.concat([normal, anomalies], ignore_index=True)
data.to_csv('test_data.csv', index=False)
```

Then run detection:

```bash
python main.py --input test_data.csv
```

## 📚 Module Documentation

### FeatureExtractor

Extracts and preprocesses features from CSV telemetry data.

```python
from gravitational_botnet_detector import FeatureExtractor

extractor = FeatureExtractor(config)
df = extractor.load_csv('data.csv')
features, names = extractor.extract_features(df)
```

### GravitationalSimulator

Implements N-Body gravitational simulation.

```python
from gravitational_botnet_detector import GravitationalSimulator

simulator = GravitationalSimulator(config)
results = simulator.simulate(features)
clusters = simulator.compute_cluster_assignments(results['final_positions'])
```

### GravitationalDetector

Main detector orchestrating the full pipeline.

```python
from gravitational_botnet_detector import GravitationalDetector

detector = GravitationalDetector(config)
results = detector.detect_from_csv('telemetry.csv')
report = detector.get_anomaly_report(results)
```

## 🛠️ Advanced Configuration

### Tuning for Accuracy

- Increase `iterations` (100-200) for better cluster convergence
- Adjust `gravitational_constant` based on feature scale
- Lower `anomaly_threshold` for higher sensitivity

### Performance Optimization

- Reduce `iterations` (20-30) for faster processing
- Use adaptive `distance_threshold` for clustering
- Sample large datasets if memory is constrained

### Domain-Specific Tuning

Different network environments may require different parameters:

- **Enterprise Network**: Higher threshold (0.75-0.85)
- **IoT Environment**: Lower threshold (0.6-0.7)
- **Cloud Services**: More iterations (75-100)

## 🤝 Contributing

Contributions are welcome! Areas for enhancement:

- Additional feature extraction methods
- Visualization of gravitational clusters
- Real-time streaming detection
- Integration with SIEM systems
- Performance optimizations

## 📄 License

This project is open source and available under the MIT License.

## 🔬 Technical Background

The gravitational model is inspired by:

- N-Body physics simulations in astrophysics
- Density-based clustering algorithms (DBSCAN, HDBSCAN)
- Force-directed graph layouts
- Swarm intelligence and particle systems

**Key Advantages**:
- Natural handling of varying cluster densities
- No need to specify number of clusters
- Interpretable physics-based model
- Robust to noise and outliers

## 📞 Support

For questions, issues, or feature requests, please open an issue on GitHub.

## 🎓 Citation

If you use this package in academic work, please cite:

```
Gravitational Botnet Detector (2024)
N-Body Physics-Inspired Anomaly Detection for Network Security
```

---

**Happy Hunting! 🔍🛡️**
