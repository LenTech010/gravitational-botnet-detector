
# 🌌 Gravitational Botnet Detector: Real-time Anomaly Detection via N-body Clustering

## 💡 Project Overview

The **Gravitational Botnet Detector** is an innovative network security system that applies principles from **computational physics** and **unsupervised machine learning** to identify coordinated malicious activity, such as botnet Command and Control (C2) communication or distributed denial-of-service (DDoS) attacks.

It models network devices as particles in a feature space. Instead of relying on signatures or simple statistical thresholds, it identifies botnets by detecting sudden, collective changes—a **"gravitational collapse"**—in device behavior. This approach makes it highly effective against **zero-day exploits** and threats hidden within **encrypted traffic**.

-----

## 🔬 Core Innovation

The system's core strength lies in its novel application of physics models to network data:

  * **N-Body Physics Simulation:** Each network device or flow session is treated as a **particle** (or "body"). Features like communication volume, frequency, and synchronicity define its abstract **mass** and **position** in a high-dimensional feature space. The system detects **collective motion anomalies** when synchronized malicious activity causes hosts to move together.
  * **Gravitational Field Modeling:** Identifies changes in the overall behavioral "field" of the network. Anomalies are flagged based on the formation of dense, tightly-coupled clusters (a "gravitational collapse"), not on packet content or known signatures.
  * **Approximate Nearest Neighbor (ANN):** Implements scalable clustering using libraries like **FAISS**. This technique efficiently approximates the computational complexity of the **Barnes-Hut optimization** ($O(N \log N)$), ensuring **real-time performance** even in large-scale enterprise networks.

-----

## 🛡️ Overall Features

  * **Real-Time & Scalable:** Designed for high-throughput, live telemetry stream processing.
  * **Zero-Day & Encrypted Traffic Resistant:** Does not rely on deep packet inspection (DPI) or known threat signatures, instead focusing on metadata correlation.
  * **Comprehensive Threat Coverage:** Capable of detecting botnets, C2 channels, DDoS activity, and suspicious lateral movement.
  * **Extensible Feature Extraction:** Easily accommodates custom network/device metrics for enhanced detection.
  * **Visualization Tools:** Includes scripts to visualize the **gravitational collapse events** and the formation of anomalous clusters.

-----

## 💻 Installation Prerequisites

Ensure your environment meets these requirements before setup:

  * **Python 3.8+**
  * **FAISS:** For high-speed Approximate Nearest Neighbor (ANN) search. *(Note: Check documentation for specific CPU/GPU prerequisites)*
  * **NumPy, Pandas:** For data manipulation and numerical computation.
  * **Scikit-learn:** For general ML utilities and preprocessing.
  * **Matplotlib:** (Optional) For data visualization.

-----

## ⚙️ SET UP and Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/YourUsername/Gravitational-Botnet-Detector.git
    cd Gravitational-Botnet-Detector
    ```

2.  **Create a Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # .\venv\Scripts\activate  # On Windows
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Quickstart Usage

Run the detector via the command line, pointing to your data and configuration:

```bash
python src/main.py --input data/telemetry_sample.csv --config config/config.yaml --output anomalies.json
```

**Or import as a module:**

```python
from src.detector import GravitationalDetector

# Initialize the detector with configuration
detector = GravitationalDetector(config="config/config.yaml")

# Run detection on a data file/stream
anomalies = detector.detect("data/live_telemetry_stream.csv")

for anomaly in anomalies:
    print(f"Anomaly Detected: {anomaly['type']} on Host: {anomaly['host']}")
```

-----

## 🧱 System Architecture

The detector operates using a modular, real-time pipeline designed for high-throughput:

1.  **Data Ingestion/Aggregator:** Ingests network flow data (NetFlow, IPFIX, or aggregated logs) from files or live streams (e.g., Kafka).
2.  **Feature Extractor:** Raw telemetry is transformed into structured **feature vectors** representing the "mass" and "position" of each host over a defined time window.
3.  **Gravitational Engine (ANN Clustering):** The FAISS-accelerated clustering engine runs in real-time to perform feature space proximity calculation, grouping hosts into clusters based on behavioral similarity.
4.  **Anomaly Scorer:** Clusters are dynamically analyzed based on metrics like **density**, **size**, and **temporal cohesion** to calculate a **Botnet Anomaly Score (BAS)**.
5.  **Alerting/Output:** Anomalies exceeding the configured threshold are logged or forwarded for security analysis.

-----

## 📈 Visualization

The project includes scripts to visualize the clustering process:

  * **Dimensionality Reduction:** Techniques (e.g., PCA, t-SNE) are used to map the high-dimensional feature space into 2D or 3D plots.
  * **Cluster Event Mapping:** Visual confirmation of the separation between dispersed normal background traffic and dense, tightly-coupled malicious clusters ("gravitational collapse events").

-----

## 🔗 Extending the Detector

The modular design facilitates easy expansion:

  * **Custom Feature Integration:** To incorporate new metrics (e.g., DNS request volume, application-level data), modify the feature extraction logic in `src/detector.py` and update the `feature_selection` list in `config/config.yaml`.
  * **New Clustering Algorithms:** The system allows for swapping out the underlying clustering method. Experiment with other density-based algorithms by adjusting the `clustering_alg` parameter in `config/config.yaml`.

-----

## 🛠️ Configuration

All operational parameters are controlled via the `config/config.yaml` file:

```yaml
# Gravitational Botnet Detector Configuration Example

detector:
  time_window_sec: 60  # Aggregate features over 60-second intervals

# Feature Engineering
features:
  selection:
    - total_bytes
    - total_packets
    - unique_dest_ips
    - port_entropy # Key features for C2 detection

# Clustering/Gravitational Engine Settings
clustering:
  algorithm: DBSCAN  # Example of density-based clustering
  params:
    eps: 0.5         # Max distance for two points to be considered neighbors
    min_samples: 10  # Minimum number of points to form a dense region

# Anomaly Scoring & Alerting
alerting:
  bas_threshold: 0.85 # Botnet Anomaly Score threshold (0.0 to 1.0)
  output_format: json
```

-----

## 📚 References and Related Works

This project draws inspiration from:

  * **Computational Physics:** The **Barnes-Hut Tree Code** and **Fast Multipole Method** for efficient $N$-body simulations.
  * **Network Security:** Research on flow-based anomaly detection and graph-based botnet analysis.

-----

## 🤝 Contributing

We welcome contributions\! Please see our `CONTRIBUTING.md` (to be created) for guidelines on submitting pull requests and reporting issues.

## 📄 License

This project is licensed under the **MIT License** (see the `LICENSE` file for details).