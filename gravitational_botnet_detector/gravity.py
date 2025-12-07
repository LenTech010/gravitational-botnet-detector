"""
Gravitational Simulation Module

Implements N-Body physics-inspired clustering for anomaly detection.
Nodes with similar behavior attract each other gravitationally, forming clusters.
Outliers experience weaker gravitational forces and remain isolated.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GravitationalSimulator:
    """
    Simulates N-Body gravitational interactions between network entities.
    
    The gravitational model:
    - Each entity (IP, connection) is treated as a mass in space
    - Similar entities attract each other (gravitational force)
    - Force proportional to mass and inversely proportional to distance squared
    - Anomalies remain isolated with weak gravitational binding
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize gravitational simulator.
        
        Args:
            config: Configuration dictionary with simulation parameters
        """
        self.config = config or {}
        self.G = self.config.get('gravitational_constant', 1.0)
        self.damping = self.config.get('damping', 0.9)
        self.iterations = self.config.get('iterations', 50)
        self.dt = self.config.get('time_step', 0.1)
        self.min_distance = self.config.get('min_distance', 0.1)
        
    def initialize_bodies(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize positions, velocities, and masses for N-Body simulation.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (positions, velocities, masses)
        """
        n_samples, n_features = features.shape
        logger.info(f"Initializing {n_samples} bodies in {n_features}-dimensional space")
        
        # Use features as initial positions (already normalized)
        positions = features.copy()
        
        # Initialize velocities to zero
        velocities = np.zeros_like(positions)
        
        # Initialize masses based on feature magnitudes
        masses = np.linalg.norm(features, axis=1) + 1.0  # Add 1 to avoid zero mass
        masses = masses / masses.mean()  # Normalize around 1.0
        
        return positions, velocities, masses
    
    def compute_gravitational_force(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        i: int,
        j: int
    ) -> np.ndarray:
        """
        Compute gravitational force between two bodies.
        
        F = G * (m1 * m2) / r^2 * direction
        
        Args:
            positions: Position matrix
            masses: Mass array
            i: Index of first body
            j: Index of second body
            
        Returns:
            Force vector
        """
        # Direction vector from i to j
        r_vec = positions[j] - positions[i]
        r_distance = np.linalg.norm(r_vec)
        
        # Avoid singularity at zero distance
        if r_distance < self.min_distance:
            r_distance = self.min_distance
        
        # Gravitational force magnitude
        force_magnitude = self.G * masses[i] * masses[j] / (r_distance ** 2)
        
        # Force vector (normalized direction * magnitude)
        force_vec = (r_vec / r_distance) * force_magnitude
        
        return force_vec
    
    def simulate(self, features: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run N-Body gravitational simulation.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Dictionary containing simulation results:
                - final_positions: Final positions after simulation
                - trajectories: Position history over iterations
                - kinetic_energy: Kinetic energy over time
                - potential_energy: Gravitational potential energy
        """
        logger.info(f"Starting gravitational simulation for {self.iterations} iterations")
        
        positions, velocities, masses = self.initialize_bodies(features)
        n_samples = len(positions)
        
        # Storage for trajectories and energies
        trajectories = [positions.copy()]
        kinetic_energies = []
        potential_energies = []
        
        for iteration in range(self.iterations):
            # Compute forces on each body
            forces = np.zeros_like(positions)
            potential_energy = 0.0
            
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    # Compute force between bodies i and j
                    force = self.compute_gravitational_force(positions, masses, i, j)
                    
                    # Newton's third law: equal and opposite forces
                    forces[i] += force
                    forces[j] -= force
                    
                    # Accumulate potential energy
                    r = np.linalg.norm(positions[j] - positions[i])
                    if r > self.min_distance:
                        potential_energy -= self.G * masses[i] * masses[j] / r
            
            # Update velocities and positions (Velocity Verlet integration)
            accelerations = forces / masses[:, np.newaxis]
            velocities += accelerations * self.dt
            velocities *= self.damping  # Apply damping
            positions += velocities * self.dt
            
            # Compute kinetic energy
            kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities ** 2)
            
            # Store trajectory and energies
            trajectories.append(positions.copy())
            kinetic_energies.append(kinetic_energy)
            potential_energies.append(potential_energy)
            
            if (iteration + 1) % 10 == 0:
                logger.debug(f"Iteration {iteration + 1}/{self.iterations}: "
                           f"KE={kinetic_energy:.4f}, PE={potential_energy:.4f}")
        
        logger.info("Gravitational simulation complete")
        
        return {
            'final_positions': positions,
            'trajectories': np.array(trajectories),
            'kinetic_energy': np.array(kinetic_energies),
            'potential_energy': np.array(potential_energies),
            'masses': masses
        }
    
    def compute_cluster_assignments(
        self,
        final_positions: np.ndarray,
        distance_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Assign bodies to clusters based on final positions.
        
        Uses simple distance-based clustering after gravitational settling.
        
        Args:
            final_positions: Final positions from simulation
            distance_threshold: Maximum distance for cluster membership
            
        Returns:
            Cluster assignment array
        """
        if distance_threshold is None:
            # Adaptive threshold based on mean pairwise distance
            distances = []
            n = len(final_positions)
            for i in range(min(n, 100)):  # Sample for efficiency
                for j in range(i + 1, min(n, 100)):
                    distances.append(np.linalg.norm(final_positions[i] - final_positions[j]))
            distance_threshold = np.median(distances) if distances else 1.0
        
        logger.info(f"Computing clusters with distance threshold: {distance_threshold:.4f}")
        
        # Simple agglomerative clustering
        n_samples = len(final_positions)
        clusters = np.arange(n_samples)  # Initially each point is its own cluster
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distance = np.linalg.norm(final_positions[i] - final_positions[j])
                if distance < distance_threshold:
                    # Merge clusters
                    clusters[clusters == clusters[j]] = clusters[i]
        
        # Renumber clusters to be contiguous
        unique_clusters = np.unique(clusters)
        cluster_map = {old: new for new, old in enumerate(unique_clusters)}
        clusters = np.array([cluster_map[c] for c in clusters])
        
        logger.info(f"Identified {len(unique_clusters)} clusters")
        
        return clusters
