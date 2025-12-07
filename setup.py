"""
Setup configuration for Gravitational Botnet Detector package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="gravitational-botnet-detector",
    version="0.1.0",
    author="Gravitational Botnet Detector Team",
    description="Real-time anomaly detection using N-Body gravitational clustering for botnets and APTs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LenTech010/gravitational-botnet-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0,<2.0.0",
        "pandas>=1.3.0,<3.0.0",
        "PyYAML>=5.4.0,<7.0.0",
    ],
    entry_points={
        "console_scripts": [
            "gravitational-detector=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
