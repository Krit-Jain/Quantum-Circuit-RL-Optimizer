"""
quantum_rl — Quantum Circuit RL Optimizer
==========================================
Reinforcement learning for quantum circuit compilation.
Trains PPO and DQN agents to discover compact gate sequences
for 2-qubit and 3-qubit target unitaries.
"""

__version__ = "1.0.0"

from quantum_rl.environment.quantum_env import QuantumCircuitEnv

__all__ = ["QuantumCircuitEnv"]