from setuptools import setup, find_packages

setup(
    name         = "quantum-rl-optimizer",
    version      = "1.0.0",
    description  = "Reinforcement learning for quantum circuit compilation",
    package_dir  = {"": "src"},
    packages     = find_packages(where="src"),
    python_requires = ">=3.10",
    install_requires = [
        "qiskit>=1.1.2",
        "qiskit-aer>=0.14.2",
        "stable-baselines3>=2.3.2",
        "gymnasium>=0.29.1",
        "torch>=2.3.1",
        "numpy>=1.26.4",
        "scipy>=1.13.1",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.2",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.4",
        "pandas>=2.2.2",
    ],
)