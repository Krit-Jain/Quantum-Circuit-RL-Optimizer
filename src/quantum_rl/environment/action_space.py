"""
action_space.py
===============
Defines the discrete action space for quantum circuit compilation.

Each action is an integer index mapping to a (gate_label, unitary_matrix) pair.
The matrix is the full n-qubit operator in the computational basis.

Gate set
--------
Fixed single-qubit  : H, X, Y, Z, S, T         (on each qubit)
Parameterized       : Rx, Ry, Rz at 4 angles    (on each qubit)
Two-qubit           : CNOT (both dirs), CZ, SWAP (all qubit pairs)

Action counts
-------------
2-qubit : 6*2 + 3*4*2 + 4*1 = 40 actions
3-qubit : 6*3 + 3*4*3 + 4*3 = 66 actions
"""

import itertools
from typing import List, Tuple
import numpy as np


class QuantumActionSpace:
    """
    Builds and stores the complete gate set as full n-qubit unitary matrices.

    Parameters
    ----------
    n_qubits : int  — number of qubits (2 or 3)
    """

    # ── base single-qubit gate matrices ──────────────────────────────────────
    _H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    _X = np.array([[0, 1], [1,  0]], dtype=complex)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    _Z = np.array([[1, 0], [0, -1]], dtype=complex)
    _S = np.array([[1, 0], [0, 1j]], dtype=complex)
    _T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    # Discrete rotation angles: π/4, π/2, 3π/4, π
    _ANGLES: List[float] = [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

    def __init__(self, n_qubits: int) -> None:
        assert n_qubits in (2, 3), "Only 2- and 3-qubit systems are supported."
        self.n_qubits = n_qubits
        self.dim      = 2 ** n_qubits

        self._actions: List[Tuple[str, np.ndarray]] = []
        self._gate_names: List[str] = []

        self._build_actions()
        self.n_actions = len(self._actions)

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def gate_names(self) -> List[str]:
        """Sorted list of unique base gate names e.g. ['CNOT','CZ','H',...]."""
        return list(self._gate_names)

    def get_gate_matrix(self, action_idx: int) -> np.ndarray:
        """Full n-qubit unitary matrix for a given action index."""
        return self._actions[action_idx][1]

    def get_action_label(self, action_idx: int) -> str:
        """Human-readable label e.g. 'CNOT_q0q1'."""
        return self._actions[action_idx][0]

    def get_base_gate_name(self, action_idx: int) -> str:
        """Base gate type for histogram bookkeeping e.g. 'CNOT'."""
        label = self._actions[action_idx][0]
        for base in self._gate_names:
            if label.startswith(base):
                return base
        return label

    def describe(self) -> None:
        """Print a summary of every action."""
        print(f"\nQuantumActionSpace — {self.n_qubits} qubits")
        print(f"Total actions : {self.n_actions}")
        print(f"Gate types    : {self._gate_names}\n")
        for i, (label, _) in enumerate(self._actions):
            print(f"  [{i:3d}]  {label}")

    # ── rotation gates ────────────────────────────────────────────────────────

    @staticmethod
    def _rx(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

    @staticmethod
    def _ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def _rz(theta: float) -> np.ndarray:
        return np.array(
            [[np.exp(-1j * theta / 2), 0],
             [0, np.exp( 1j * theta / 2)]],
            dtype=complex,
        )

    # ── embedding helpers ─────────────────────────────────────────────────────

    def _embed_single(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """
        Embed a single-qubit gate into the n-qubit space via tensor product.
        Convention: qubit 0 = most significant bit (leftmost).
        Gate on qubit k → I⊗…⊗G⊗…⊗I  (G at position k).
        """
        ops = [np.eye(2, dtype=complex)] * self.n_qubits
        ops[qubit] = gate
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def _cnot(self, control: int, target: int) -> np.ndarray:
        """
        CNOT: flip target bit when control bit is |1⟩.
        Big-endian bit ordering — qubit 0 is MSB.
        """
        mat = np.zeros((self.dim, self.dim), dtype=complex)
        for col in range(self.dim):
            bits = list(format(col, f"0{self.n_qubits}b"))
            if bits[control] == "1":
                bits[target] = "0" if bits[target] == "1" else "1"
            row = int("".join(bits), 2)
            mat[row, col] = 1.0
        return mat

    def _cz(self, q0: int, q1: int) -> np.ndarray:
        """CZ: apply −1 phase when both qubits are |1⟩."""
        mat = np.eye(self.dim, dtype=complex)
        for i in range(self.dim):
            bits = format(i, f"0{self.n_qubits}b")
            if bits[q0] == "1" and bits[q1] == "1":
                mat[i, i] = -1.0
        return mat

    def _swap(self, q0: int, q1: int) -> np.ndarray:
        """SWAP: exchange amplitudes of qubits q0 and q1."""
        mat = np.zeros((self.dim, self.dim), dtype=complex)
        for col in range(self.dim):
            bits = list(format(col, f"0{self.n_qubits}b"))
            bits[q0], bits[q1] = bits[q1], bits[q0]
            row = int("".join(bits), 2)
            mat[row, col] = 1.0
        return mat

    # ── action builder ────────────────────────────────────────────────────────

    def _build_actions(self) -> None:
        gate_name_set = set()

        # fixed single-qubit gates
        fixed = [
            ("H", self._H), ("X", self._X), ("Y", self._Y),
            ("Z", self._Z), ("S", self._S), ("T", self._T),
        ]
        for qubit in range(self.n_qubits):
            for name, gate in fixed:
                mat = self._embed_single(gate, qubit)
                self._actions.append((f"{name}_q{qubit}", mat))
                gate_name_set.add(name)

        # parameterized single-qubit gates
        param = [("Rx", self._rx), ("Ry", self._ry), ("Rz", self._rz)]
        for qubit in range(self.n_qubits):
            for name, fn in param:
                for angle in self._ANGLES:
                    mat  = self._embed_single(fn(angle), qubit)
                    frac = f"{angle / np.pi:.2f}pi"
                    self._actions.append((f"{name}_{frac}_q{qubit}", mat))
                    gate_name_set.add(name)

        # two-qubit gates on all qubit pairs
        for q0, q1 in itertools.combinations(range(self.n_qubits), 2):
            self._actions.append((f"CNOT_q{q0}q{q1}", self._cnot(q0, q1)))
            self._actions.append((f"CNOT_q{q1}q{q0}", self._cnot(q1, q0)))
            self._actions.append((f"CZ_q{q0}q{q1}",   self._cz(q0, q1)))
            self._actions.append((f"SWAP_q{q0}q{q1}", self._swap(q0, q1)))
            gate_name_set.update(["CNOT", "CZ", "SWAP"])

        self._gate_names = sorted(gate_name_set)