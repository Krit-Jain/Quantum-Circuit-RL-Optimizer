"""
curriculum.py
=============
Automatic difficulty scheduler for progressive training.

Difficulty controls how "far from identity" the target unitary is.
At difficulty=0.0 the target is near-identity (trivial to solve).
At difficulty=1.0 the target is a fully Haar-random unitary (hardest).

The schedule
------------
  U_target = expm(i * difficulty * pi * H)

where H is a random Hermitian matrix.  As difficulty → 0, U_target → I.

Auto-scaling
------------
After every `eval_interval` episodes, if the agent's success rate
is above `promote_threshold`  → difficulty increases by `delta`
if the success rate is below  `demote_threshold`  → difficulty decreases
This keeps the agent in the "zone of proximal development".
"""

from typing import Optional
import numpy as np
from scipy.linalg import expm


class CurriculumScheduler:
    """
    Manages progressive difficulty for target unitary sampling.

    Parameters
    ----------
    n_qubits           : int
    initial_difficulty : float   Starting difficulty in [0, 1]  (default 0.1)
    min_difficulty     : float   Floor (default 0.05)
    max_difficulty     : float   Ceiling (default 1.0)
    delta              : float   Step size per promotion/demotion (default 0.05)
    promote_threshold  : float   Success rate above which we promote (default 0.7)
    demote_threshold   : float   Success rate below which we demote  (default 0.3)
    eval_interval      : int     Episodes between difficulty updates  (default 20)
    seed               : int     Optional RNG seed
    """

    def __init__(
        self,
        n_qubits:           int   = 2,
        initial_difficulty: float = 0.1,
        min_difficulty:     float = 0.05,
        max_difficulty:     float = 1.0,
        delta:              float = 0.05,
        promote_threshold:  float = 0.7,
        demote_threshold:   float = 0.3,
        eval_interval:      int   = 20,
        seed:               Optional[int] = None,
    ) -> None:
        self.n_qubits     = n_qubits
        self.dim          = 2 ** n_qubits
        self.difficulty   = float(initial_difficulty)
        self.min_d        = float(min_difficulty)
        self.max_d        = float(max_difficulty)
        self.delta        = float(delta)
        self.promote_thr  = promote_threshold
        self.demote_thr   = demote_threshold
        self.eval_interval= eval_interval
        self._rng         = np.random.default_rng(seed)

        # rolling window for success tracking
        self._recent_successes: list = []
        self._episode_count: int = 0

        # history for plotting
        self.difficulty_history: list = []
        self.success_rate_history: list = []

    # ── public API ────────────────────────────────────────────────────────────

    def sample_target(self) -> np.ndarray:
        """
        Sample a target unitary at the current difficulty level.

        Returns
        -------
        np.ndarray  shape=(dim, dim)  dtype=complex
        """
        return self._make_unitary(self.difficulty)

    def record_episode(self, success: bool) -> None:
        """
        Call at the end of each episode with whether the agent succeeded.
        Internally updates the success window and triggers difficulty update.
        """
        self._recent_successes.append(int(success))
        self._episode_count += 1

        # keep only the last `eval_interval` episodes
        if len(self._recent_successes) > self.eval_interval:
            self._recent_successes.pop(0)

        # update difficulty every eval_interval episodes
        if self._episode_count % self.eval_interval == 0:
            self._update_difficulty()

    def get_stats(self) -> dict:
        """Current curriculum statistics."""
        rate = (
            float(np.mean(self._recent_successes))
            if self._recent_successes else 0.0
        )
        return {
            "difficulty":    round(self.difficulty, 4),
            "success_rate":  round(rate, 4),
            "episode_count": self._episode_count,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _update_difficulty(self) -> None:
        if not self._recent_successes:
            return

        rate = float(np.mean(self._recent_successes))
        self.success_rate_history.append(rate)
        self.difficulty_history.append(self.difficulty)

        if rate >= self.promote_thr:
            self.difficulty = min(self.difficulty + self.delta, self.max_d)
        elif rate <= self.demote_thr:
            self.difficulty = max(self.difficulty - self.delta, self.min_d)
        # else: stay at current difficulty

    def _make_unitary(self, difficulty: float) -> np.ndarray:
        """
        Generate a unitary at given difficulty via matrix exponentiation.

        U = expm(i * difficulty * pi * H)
        where H is a random Hermitian matrix with ||H||_F ~ 1.

        At difficulty=0 → U = I
        At difficulty=1 → U is a generic element of SU(d)
        """
        # random Hermitian H = (A + A†) / 2, normalised
        A = self._rng.standard_normal((self.dim, self.dim)) + \
            1j * self._rng.standard_normal((self.dim, self.dim))
        H = (A + A.conj().T) / 2.0

        # normalise so ||H||_F = 1 regardless of dim
        norm = np.linalg.norm(H, "fro")
        if norm > 1e-12:
            H /= norm

        U = expm(1j * difficulty * np.pi * H)
        return U.astype(complex)