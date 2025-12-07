"""
Main CLI for Gravitational Botnet Detector

Usage:
    python main.py --input data.csv --config config.yaml
    python main.py --input data.csv --threshold 0.8 --iterations 100
"""

import argparse
import yaml
import sys
import logging
from pathlib import Path

from gravitational_botnet_detector import GravitationalDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Gravitational Botnet Detector - Real-time anomaly detection using N-Body physics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input telemetry.csv
  python main.py --input data.csv --config config.yaml
  python main.py --input data.csv --threshold 0.8 --iterations 100
  python main.py --input data.csv --output results.txt
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file with telemetry data'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config.yaml',
        help='Configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file for results (default: print to stdout)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        help='Anomaly score threshold (0-1, overrides config)'
    )
    
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of simulation iterations (overrides config)'
    )
    
    parser.add_argument(
        '--gravitational-constant', '-G',
        type=float,
        help='Gravitational constant (overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except results'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Check input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Load configuration
    config = {}
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
    
    # Override config with command line arguments
    if args.threshold is not None:
        config['anomaly_threshold'] = args.threshold
    
    if args.iterations is not None:
        if 'gravity' not in config:
            config['gravity'] = {}
        config['gravity']['iterations'] = args.iterations
    
    if args.gravitational_constant is not None:
        if 'gravity' not in config:
            config['gravity'] = {}
        config['gravity']['gravitational_constant'] = args.gravitational_constant
    
    # Initialize detector
    logger.info("Initializing Gravitational Botnet Detector")
    detector = GravitationalDetector(config)
    
    # Run detection
    try:
        logger.info(f"Processing input file: {args.input}")
        results = detector.detect_from_csv(args.input)
        
        # Generate report
        report = detector.get_anomaly_report(results)
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Results written to {args.output}")
        else:
            print("\n" + report)
        
        # Exit with appropriate code
        n_anomalies = results['anomalies'].sum()
        if n_anomalies > 0:
            logger.warning(f"⚠️  {n_anomalies} anomalies detected!")
            sys.exit(1)  # Non-zero exit if anomalies found
        else:
            logger.info("✓ No anomalies detected")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Error during detection: {e}", exc_info=True)
        sys.exit(2)


if __name__ == '__main__':
    main()
