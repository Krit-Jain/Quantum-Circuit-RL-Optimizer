"""
state_encoder.py
================
Converts the raw MDP state into a fixed-length float32 observation vector
for the RL agent.

Observation layout  (d = 2^n_qubits)
--------------------------------------
 Component            Size        Range    Notes
 ──────────────────── ──────────  ──────── ──────────────────────────────
 current_U  (real)    d²          [-1, 1]  Flattened real parts
 current_U  (imag)    d²          [-1, 1]  Flattened imaginary parts
 target_U   (real)    d²          [-1, 1]  Flattened real parts
 target_U   (imag)    d²          [-1, 1]  Flattened imaginary parts
 frobenius_dist       1           [0, 1]   Normalised distance to target
 process_fidelity     1           [0, 1]   F(U_current, U_target)
 normalised_depth     1           [0, 1]   step_count / max_steps
 gate_histogram       n_types     [0, 1]   Per-gate usage, normalised
 ──────────────────── ──────────  ──────── ──────────────────────────────

 2-qubit total :  2*16 + 2*16 + 3 + 12  =  79  floats
 3-qubit total :  2*64 + 2*64 + 3 + 12  =  271 floats
"""

from typing import Dict, List
import numpy as np


class StateEncoder:
    """
    Encodes MDP state into a flat float32 observation array.

    Parameters
    ----------
    n_qubits   : int
    max_steps  : int        Episode length cap (used for normalisation).
    gate_names : List[str]  Sorted list of unique base gate names from
                            QuantumActionSpace.gate_names
    """

    def __init__(
        self,
        n_qubits:   int,
        max_steps:  int,
        gate_names: List[str],
    ) -> None:
        self.n_qubits     = n_qubits
        self.dim          = 2 ** n_qubits
        self.max_steps    = max_steps
        self.gate_names   = gate_names
        self.n_gate_types = len(gate_names)

        self._gate_to_idx: Dict[str, int] = {
            name: i for i, name in enumerate(gate_names)
        }

        # precompute obs dimension so Gymnasium can use it at __init__ time
        mat_flat     = 2 * (self.dim ** 2)   # real + imag for ONE unitary
        self.obs_dim = (
            2 * mat_flat        # current_U + target_U
            + 3                 # frob_dist, fidelity, norm_depth
            + self.n_gate_types # gate histogram
        )

        # max possible Frobenius distance between two unitaries:
        # ||A - B||_F  <=  ||A||_F + ||B||_F  =  sqrt(d) + sqrt(d)
        self._max_frob = 2.0 * np.sqrt(float(self.dim))

    # ── public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        current_unitary: np.ndarray,
        target_unitary:  np.ndarray,
        fidelity:        float,
        circuit_depth:   int,
        gate_counts:     Dict[str, int],
    ) -> np.ndarray:
        """
        Build and return the flat observation vector.

        Returns
        -------
        np.ndarray  shape=(obs_dim,)  dtype=float32
        """
        obs = np.concatenate([
            self._flatten_unitary(current_unitary),
            self._flatten_unitary(target_unitary),
            self._scalar_features(current_unitary, target_unitary,
                                  fidelity, circuit_depth),
            self._gate_histogram(gate_counts),
        ])
        return obs.astype(np.float32)

    # ── private helpers ───────────────────────────────────────────────────────

    def _flatten_unitary(self, U: np.ndarray) -> np.ndarray:
        """
        Flatten complex d×d matrix to 2*d² real vector [real..., imag...].
        Unitary entries satisfy |U_ij| <= 1 so values are already in [-1, 1].
        """
        return np.concatenate([U.real.ravel(), U.imag.ravel()])

    def _scalar_features(
        self,
        current_U: np.ndarray,
        target_U:  np.ndarray,
        fidelity:  float,
        depth:     int,
    ) -> np.ndarray:
        frob       = float(np.linalg.norm(current_U - target_U, "fro"))
        norm_frob  = np.clip(frob / self._max_frob, 0.0, 1.0)
        norm_fid   = np.clip(float(fidelity),        0.0, 1.0)
        norm_depth = np.clip(depth / self.max_steps, 0.0, 1.0)
        return np.array([norm_frob, norm_fid, norm_depth], dtype=np.float64)

    def _gate_histogram(self, gate_counts: Dict[str, int]) -> np.ndarray:
        total = sum(gate_counts.values()) + 1e-8   # avoid div-by-zero
        hist  = np.zeros(self.n_gate_types, dtype=np.float32)
        for name, count in gate_counts.items():
            if name in self._gate_to_idx:
                hist[self._gate_to_idx[name]] = count / total
        return hist