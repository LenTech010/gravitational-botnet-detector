"""
Demo script showing how to use the Gravitational Botnet Detector Python API.

This demonstrates:
1. Loading and processing CSV data
2. Running detection with custom configuration
3. Accessing and analyzing results
4. Generating reports
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gravitational_botnet_detector import (
    GravitationalDetector,
    FeatureExtractor,
    GravitationalSimulator
)

def demo_basic_usage():
    """Basic usage example."""
    print("=" * 70)
    print("DEMO 1: Basic Usage")
    print("=" * 70)
    
    # Initialize detector with default config
    detector = GravitationalDetector()
    
    # Run detection on sample data
    results = detector.detect_from_csv('data/sample_telemetry.csv')
    
    # Print report
    report = detector.get_anomaly_report(results)
    print(report)
    
    print("\n")


def demo_custom_config():
    """Example with custom configuration."""
    print("=" * 70)
    print("DEMO 2: Custom Configuration")
    print("=" * 70)
    
    # Custom configuration
    config = {
        'anomaly_threshold': 0.6,
        'gravity': {
            'iterations': 75,
            'gravitational_constant': 1.5,
            'damping': 0.85
        }
    }
    
    detector = GravitationalDetector(config)
    results = detector.detect_from_csv('data/sample_telemetry.csv')
    
    # Access specific results
    n_anomalies = results['anomalies'].sum()
    n_clusters = len(set(results['clusters']))
    
    print(f"Configuration used:")
    print(f"  - Threshold: {config['anomaly_threshold']}")
    print(f"  - Iterations: {config['gravity']['iterations']}")
    print(f"  - Gravitational constant: {config['gravity']['gravitational_constant']}")
    print(f"\nResults:")
    print(f"  - Total samples: {len(results['anomalies'])}")
    print(f"  - Anomalies detected: {n_anomalies} ({100*n_anomalies/len(results['anomalies']):.1f}%)")
    print(f"  - Clusters formed: {n_clusters}")
    
    # Show top 5 anomaly scores
    import numpy as np
    top_indices = np.argsort(results['anomaly_scores'])[-5:][::-1]
    print(f"\nTop 5 anomaly scores:")
    for idx in top_indices:
        score = results['anomaly_scores'][idx]
        cluster = results['clusters'][idx]
        print(f"  Sample {idx}: score={score:.4f}, cluster={cluster}")
    
    print("\n")


def demo_modular_usage():
    """Example showing modular usage of individual components."""
    print("=" * 70)
    print("DEMO 3: Modular Usage")
    print("=" * 70)
    
    # Step 1: Feature extraction
    print("Step 1: Feature Extraction")
    extractor = FeatureExtractor()
    df = extractor.load_csv('data/sample_telemetry.csv')
    features, feature_names = extractor.extract_features(df)
    print(f"  Extracted {features.shape[1]} features from {features.shape[0]} samples")
    
    # Step 2: Gravitational simulation
    print("\nStep 2: Gravitational Simulation")
    simulator = GravitationalSimulator({
        'iterations': 50,
        'gravitational_constant': 1.0
    })
    sim_results = simulator.simulate(features)
    print(f"  Simulation complete: {len(sim_results['trajectories'])} iterations")
    print(f"  Final kinetic energy: {sim_results['kinetic_energy'][-1]:.4f}")
    print(f"  Final potential energy: {sim_results['potential_energy'][-1]:.4f}")
    
    # Step 3: Cluster analysis
    print("\nStep 3: Cluster Analysis")
    clusters = simulator.compute_cluster_assignments(sim_results['final_positions'])
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    print(f"  Number of clusters: {len(unique_clusters)}")
    print(f"  Cluster sizes: {dict(zip(unique_clusters, counts))}")
    
    print("\n")


def demo_analysis():
    """Example showing detailed analysis of results."""
    print("=" * 70)
    print("DEMO 4: Detailed Analysis")
    print("=" * 70)
    
    detector = GravitationalDetector()
    results = detector.detect_from_csv('data/sample_telemetry.csv')
    
    # Analyze cluster sizes
    import numpy as np
    clusters = results['clusters']
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    
    print("Cluster Size Distribution:")
    print(f"  Total clusters: {len(unique_clusters)}")
    print(f"  Average cluster size: {counts.mean():.2f}")
    print(f"  Largest cluster: {counts.max()} samples")
    print(f"  Smallest cluster: {counts.min()} samples")
    
    # Analyze score distribution
    scores = results['anomaly_scores']
    print(f"\nAnomaly Score Statistics:")
    print(f"  Mean: {scores.mean():.4f}")
    print(f"  Std: {scores.std():.4f}")
    print(f"  Min: {scores.min():.4f}")
    print(f"  Max: {scores.max():.4f}")
    print(f"  Median: {np.median(scores):.4f}")
    
    # Count by threshold
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\nAnomalies by threshold:")
    for threshold in thresholds:
        count = (scores > threshold).sum()
        print(f"  > {threshold}: {count} samples ({100*count/len(scores):.1f}%)")
    
    print("\n")


if __name__ == '__main__':
    import numpy as np
    
    print("\n" + "=" * 70)
    print("GRAVITATIONAL BOTNET DETECTOR - API DEMO")
    print("=" * 70 + "\n")
    
    # Run all demos
    demo_basic_usage()
    demo_custom_config()
    demo_modular_usage()
    demo_analysis()
    
    print("=" * 70)
    print("Demo complete! Check the code in demo.py for implementation details.")
    print("=" * 70)
