"""
Anomaly Detection Module

Integrates feature extraction and gravitational simulation for botnet detection.
Computes anomaly scores based on gravitational binding and cluster isolation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

from .features import FeatureExtractor
from .gravity import GravitationalSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GravitationalDetector:
    """
    Main detector class that orchestrates feature extraction, gravitational
    simulation, and anomaly scoring for botnet detection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the gravitational detector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_extractor = FeatureExtractor(self.config.get('features', {}))
        self.gravity_simulator = GravitationalSimulator(self.config.get('gravity', {}))
        
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.7)
        self.cluster_size_weight = self.config.get('cluster_size_weight', 0.3)
        
    def detect_from_csv(self, csv_path: str) -> Dict:
        """
        Detect anomalies from CSV telemetry file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Dictionary containing detection results
        """
        logger.info(f"Starting anomaly detection for {csv_path}")
        
        # Step 1: Load and extract features
        df = self.feature_extractor.load_csv(csv_path)
        features, feature_names = self.feature_extractor.extract_features(df)
        
        # Step 2: Run gravitational simulation
        sim_results = self.gravity_simulator.simulate(features)
        
        # Step 3: Compute cluster assignments
        clusters = self.gravity_simulator.compute_cluster_assignments(
            sim_results['final_positions']
        )
        
        # Step 4: Compute anomaly scores
        anomaly_scores = self.compute_anomaly_scores(
            sim_results['final_positions'],
            clusters,
            sim_results['masses']
        )
        
        # Step 5: Identify anomalies
        anomalies = anomaly_scores > self.anomaly_threshold
        
        logger.info(f"Detected {anomalies.sum()} anomalies out of {len(anomalies)} samples")
        
        return {
            'anomaly_scores': anomaly_scores,
            'anomalies': anomalies,
            'clusters': clusters,
            'features': features,
            'feature_names': feature_names,
            'simulation_results': sim_results,
            'dataframe': df
        }
    
    def compute_anomaly_scores(
        self,
        positions: np.ndarray,
        clusters: np.ndarray,
        masses: np.ndarray
    ) -> np.ndarray:
        """
        Compute anomaly scores based on gravitational properties.
        
        Anomaly indicators:
        1. Distance from cluster center (isolation)
        2. Small cluster size (unusual group)
        3. Low gravitational binding energy
        4. High kinetic energy (unstable)
        
        Args:
            positions: Final positions from simulation
            clusters: Cluster assignments
            masses: Mass values
            
        Returns:
            Array of anomaly scores [0, 1]
        """
        n_samples = len(positions)
        scores = np.zeros(n_samples)
        
        # Compute cluster properties
        unique_clusters = np.unique(clusters)
        cluster_centers = {}
        cluster_sizes = {}
        
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_centers[cluster_id] = positions[mask].mean(axis=0)
            cluster_sizes[cluster_id] = mask.sum()
        
        # Compute scores for each sample
        for i in range(n_samples):
            cluster_id = clusters[i]
            
            # 1. Distance from cluster center (normalized)
            center = cluster_centers[cluster_id]
            distance_from_center = np.linalg.norm(positions[i] - center)
            
            # 2. Cluster size penalty (small clusters are suspicious)
            cluster_size = cluster_sizes[cluster_id]
            size_score = 1.0 / (1.0 + np.log(cluster_size + 1))
            
            # 3. Isolation score (distance to nearest neighbor)
            distances = np.linalg.norm(positions - positions[i], axis=1)
            distances[i] = np.inf  # Exclude self
            nearest_distance = distances.min()
            
            # 4. Gravitational binding (average force with cluster members)
            cluster_mask = clusters == cluster_id
            cluster_positions = positions[cluster_mask]
            cluster_masses = masses[cluster_mask]
            
            binding_energy = 0.0
            if len(cluster_positions) > 1:
                for j, (pos, mass) in enumerate(zip(cluster_positions, cluster_masses)):
                    if not np.allclose(pos, positions[i]):
                        r = np.linalg.norm(pos - positions[i])
                        if r > 0.1:
                            binding_energy += mass / r
                binding_energy /= len(cluster_positions)
            
            # Combine scores (higher = more anomalous)
            isolation_score = distance_from_center + nearest_distance
            weak_binding_score = 1.0 / (1.0 + binding_energy)
            
            # Weighted combination
            score = (
                0.3 * isolation_score +
                0.3 * size_score +
                0.4 * weak_binding_score
            )
            
            scores[i] = score
        
        # Normalize scores to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        
        logger.info(f"Computed anomaly scores: min={scores.min():.4f}, "
                   f"max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        return scores
    
    def get_anomaly_report(self, results: Dict) -> str:
        """
        Generate a human-readable anomaly detection report.
        
        Args:
            results: Detection results from detect_from_csv
            
        Returns:
            Formatted report string
        """
        anomaly_indices = np.where(results['anomalies'])[0]
        n_total = len(results['anomalies'])
        n_anomalies = len(anomaly_indices)
        
        report = []
        report.append("=" * 60)
        report.append("GRAVITATIONAL BOTNET DETECTOR - ANOMALY REPORT")
        report.append("=" * 60)
        report.append(f"Total Samples: {n_total}")
        report.append(f"Anomalies Detected: {n_anomalies} ({100*n_anomalies/n_total:.1f}%)")
        report.append(f"Number of Clusters: {len(np.unique(results['clusters']))}")
        report.append("-" * 60)
        
        if n_anomalies > 0:
            report.append("ANOMALOUS SAMPLES:")
            report.append(f"{'Index':<10} {'Score':<10} {'Cluster':<10}")
            report.append("-" * 60)
            
            for idx in anomaly_indices[:20]:  # Show top 20
                score = results['anomaly_scores'][idx]
                cluster = results['clusters'][idx]
                report.append(f"{idx:<10} {score:<10.4f} {cluster:<10}")
            
            if n_anomalies > 20:
                report.append(f"... and {n_anomalies - 20} more anomalies")
        else:
            report.append("No anomalies detected.")
        
        report.append("=" * 60)
        
        return "\n".join(report)
