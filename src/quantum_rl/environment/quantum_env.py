"""
quantum_env.py
==============
Gymnasium MDP environment for quantum circuit compilation via RL.

The agent's goal: find a gate sequence that implements a target unitary
U_target with process fidelity >= fidelity_threshold while keeping
circuit depth as low as possible.

MDP formulation
---------------
  State   : [current_U, target_U, fidelity, depth, gate_histogram]
              (see StateEncoder for exact layout)
  Action  : Discrete gate selection (see QuantumActionSpace)
  Reward  : 3-tier — dense progress + potential shaping + sparse terminal

Reward strategies (selectable via reward_type)
-----------------------------------------------
  "binary"  — sparse only: +100 at success, −depth_penalty per step
  "dense"   — delta-fidelity * 10 + terminal bonus
  "log"     — −log(1 − F) potential + terminal bonus
  "hybrid"  — dense + log-potential + terminal  ← recommended
"""

from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from qiskit.quantum_info import random_unitary

from quantum_rl.environment.action_space  import QuantumActionSpace
from quantum_rl.environment.state_encoder import StateEncoder


class QuantumCircuitEnv(gym.Env):
    """
    Parameters
    ----------
    n_qubits            : int    2 or 3
    max_steps           : int    Maximum gates per episode
    fidelity_threshold  : float  Success criterion (default 0.99)
    reward_type         : str    One of binary | dense | log | hybrid
    depth_penalty       : float  Per-step penalty to encourage short circuits
    seed                : int    Optional RNG seed
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_qubits:           int   = 2,
        max_steps:          int   = 50,
        fidelity_threshold: float = 0.99,
        reward_type:        str   = "hybrid",
        depth_penalty:      float = 0.01,
        seed:               Optional[int] = None,
    ) -> None:
        super().__init__()

        assert n_qubits in (2, 3), "Only 2- or 3-qubit systems supported."
        assert reward_type in ("binary", "dense", "log", "hybrid"), \
            f"Unknown reward_type '{reward_type}'."

        self.n_qubits           = n_qubits
        self.dim                = 2 ** n_qubits
        self.max_steps          = max_steps
        self.fidelity_threshold = fidelity_threshold
        self.reward_type        = reward_type
        self.depth_penalty      = depth_penalty

        # ── action & state components ─────────────────────────────────────────
        self.action_handler = QuantumActionSpace(n_qubits)
        self.state_encoder  = StateEncoder(
            n_qubits  = n_qubits,
            max_steps = max_steps,
            gate_names= self.action_handler.gate_names,
        )

        # ── Gymnasium spaces ──────────────────────────────────────────────────
        self.action_space = spaces.Discrete(self.action_handler.n_actions)
        self.observation_space = spaces.Box(
            low   = -1.0,
            high  =  1.0,
            shape = (self.state_encoder.obs_dim,),
            dtype = np.float32,
        )

        # ── episode state (populated by reset()) ─────────────────────────────
        self.target_unitary:  Optional[np.ndarray] = None
        self.current_unitary: Optional[np.ndarray] = None
        self.step_count:      int   = 0
        self.circuit_depth:   int   = 0
        self.prev_fidelity:   float = 0.0
        self.gate_counts:     Dict[str, int] = {}

        # ── RNG ───────────────────────────────────────────────────────────────
        self._seed = seed
        self._rng  = np.random.default_rng(seed)

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium core API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # allow caller to inject a specific target (useful for curriculum)
        if options and "target_unitary" in options:
            self.target_unitary = np.array(options["target_unitary"], dtype=complex)
        else:
            self.target_unitary = self._sample_target_unitary()

        # start from identity circuit
        self.current_unitary = np.eye(self.dim, dtype=complex)
        self.step_count      = 0
        self.circuit_depth   = 0
        self.gate_counts     = {name: 0 for name in self.action_handler.gate_names}
        self.prev_fidelity   = self._compute_fidelity()

        obs  = self._get_obs()
        info = {"fidelity": self.prev_fidelity, "depth": self.circuit_depth}
        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.target_unitary is not None, "Call reset() before step()."

        # ── apply gate ────────────────────────────────────────────────────────
        gate_matrix = self.action_handler.get_gate_matrix(action)
        gate_name   = self.action_handler.get_base_gate_name(action)

        # U_new = G @ U_current  (gate G applied on top of current circuit)
        self.current_unitary = gate_matrix @ self.current_unitary
        self.circuit_depth  += 1
        self.step_count     += 1
        self.gate_counts[gate_name] = self.gate_counts.get(gate_name, 0) + 1

        # ── evaluate ──────────────────────────────────────────────────────────
        fidelity   = self._compute_fidelity()
        reward     = self._compute_reward(fidelity)
        terminated = bool(fidelity >= self.fidelity_threshold)
        truncated  = bool(self.step_count >= self.max_steps)

        self.prev_fidelity = fidelity

        obs  = self._get_obs()
        info = {
            "fidelity":    fidelity,
            "depth":       self.circuit_depth,
            "gate_counts": dict(self.gate_counts),
            "success":     terminated,
        }
        return obs, reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        return self.state_encoder.encode(
            current_unitary = self.current_unitary,
            target_unitary  = self.target_unitary,
            fidelity        = self.prev_fidelity,
            circuit_depth   = self.circuit_depth,
            gate_counts     = self.gate_counts,
        )

    def _compute_fidelity(self) -> float:
        """
        Process fidelity between current circuit unitary and target unitary.

        F = |Tr(U_target† @ U_current)|² / d²

        Properties:
          - F = 1  when U_current == U_target  (up to global phase)
          - F = 0  for maximally orthogonal unitaries
          - Phase-invariant: F(e^{iφ}U, U) = 1
        """
        d   = self.dim
        hs  = np.trace(self.target_unitary.conj().T @ self.current_unitary)
        f   = float(np.abs(hs) ** 2) / float(d * d)
        return float(np.clip(f, 0.0, 1.0))

    def _compute_reward(self, fidelity: float) -> float:
        """
        3-tier reward function.

        Tier 1 (all modes) — per-step depth penalty encourages short circuits.
        Tier 2             — dense or potential-based shaping per step.
        Tier 3             — large sparse bonus on success.
        """
        delta_f  = fidelity - self.prev_fidelity
        eps      = 1e-8
        terminal = 100.0 if fidelity >= self.fidelity_threshold else 0.0

        if self.reward_type == "binary":
            return terminal - self.depth_penalty

        elif self.reward_type == "dense":
            return delta_f * 10.0 + terminal - self.depth_penalty

        elif self.reward_type == "log":
            # −log(1−F) → strongly rewards progress near F=1 where
            # delta_F becomes tiny but the agent still needs guidance
            log_reward = -np.log(1.0 - fidelity + eps)
            return log_reward + terminal - self.depth_penalty

        elif self.reward_type == "hybrid":
            # Dense progress signal  (fast early learning)
            dense     = delta_f * 10.0
            # Log-potential shaping  (fine-grained signal near F=1)
            potential = 0.5 * (-np.log(1.0 - fidelity + eps))
            return dense + potential + terminal - self.depth_penalty

        # unreachable — caught in __init__
        raise ValueError(f"Unknown reward_type: {self.reward_type}")

    def _sample_target_unitary(self) -> np.ndarray:
        """Sample a Haar-random unitary from SU(2^n_qubits)."""
        u = random_unitary(self.dim)
        return np.array(u.data, dtype=complex)

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def get_circuit_stats(self) -> Dict:
        """Snapshot of current episode statistics — useful for logging."""
        return {
            "fidelity":   self._compute_fidelity(),
            "depth":      self.circuit_depth,
            "gate_counts":dict(self.gate_counts),
            "total_gates":sum(self.gate_counts.values()),
        }