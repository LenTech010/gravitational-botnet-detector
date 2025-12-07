"""
Generate sample telemetry data for testing the Gravitational Botnet Detector.

This script creates synthetic network telemetry with:
- Normal traffic patterns (majority)
- Botnet-like anomalies (coordinated, high volume)
- Random anomalies (outliers)
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate normal traffic (80 samples)
print("Generating normal traffic patterns...")
normal_traffic = pd.DataFrame({
    'timestamp': np.arange(1700000000, 1700000000 + 80),
    'bytes_sent': np.random.normal(1500, 300, 80).astype(int),
    'bytes_received': np.random.normal(3000, 600, 80).astype(int),
    'packets': np.random.poisson(15, 80),
    'duration': np.random.exponential(2.0, 80),
    'connections': np.random.poisson(3, 80),
})

# Generate botnet traffic (15 samples) - coordinated, high volume
print("Generating botnet-like anomalies...")
botnet_traffic = pd.DataFrame({
    'timestamp': np.arange(1700000080, 1700000080 + 15),
    'bytes_sent': np.random.normal(8000, 500, 15).astype(int),
    'bytes_received': np.random.normal(15000, 1000, 15).astype(int),
    'packets': np.random.poisson(60, 15),
    'duration': np.random.exponential(0.8, 15),
    'connections': np.random.poisson(12, 15),
})

# Generate random outliers (5 samples)
print("Generating random outliers...")
outliers = pd.DataFrame({
    'timestamp': np.arange(1700000095, 1700000095 + 5),
    'bytes_sent': np.random.uniform(100, 20000, 5).astype(int),
    'bytes_received': np.random.uniform(200, 30000, 5).astype(int),
    'packets': np.random.randint(1, 100, 5),
    'duration': np.random.uniform(0.1, 10.0, 5),
    'connections': np.random.randint(1, 20, 5),
})

# Combine all traffic
all_traffic = pd.concat([normal_traffic, botnet_traffic, outliers], ignore_index=True)

# Shuffle the data
all_traffic = all_traffic.sample(frac=1).reset_index(drop=True)

# Add some derived features
all_traffic['bytes_total'] = all_traffic['bytes_sent'] + all_traffic['bytes_received']
all_traffic['bytes_ratio'] = all_traffic['bytes_sent'] / (all_traffic['bytes_received'] + 1)
all_traffic['packets_per_second'] = all_traffic['packets'] / (all_traffic['duration'] + 0.1)

# Save to CSV
output_file = 'data/sample_telemetry.csv'
all_traffic.to_csv(output_file, index=False)

print(f"\n✓ Generated {len(all_traffic)} samples:")
print(f"  - Normal traffic: 80 samples")
print(f"  - Botnet anomalies: 15 samples")
print(f"  - Random outliers: 5 samples")
print(f"\n✓ Saved to: {output_file}")
print(f"\nColumns: {', '.join(all_traffic.columns)}")
print(f"\nFirst few rows:")
print(all_traffic.head(10))
