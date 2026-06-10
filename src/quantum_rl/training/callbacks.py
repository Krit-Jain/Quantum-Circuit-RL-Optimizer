"""
callbacks.py
============
Custom Stable-Baselines3 callbacks for training monitoring.

FidelityCallback   — logs mean fidelity & success rate during training
CurriculumCallback — hooks into the curriculum scheduler at episode end
"""

from typing import Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class FidelityCallback(BaseCallback):
    """
    Logs episode-level fidelity and success rate to SB3's logger.

    Metrics logged (visible in tensorboard or stdout)
    -------------------------------------------------
    train/mean_fidelity   — mean final fidelity over last N episodes
    train/success_rate    — fraction of episodes where F >= threshold
    train/mean_depth      — mean circuit depth on episode end
    """

    def __init__(
        self,
        window_size: int = 100,
        verbose:     int = 0,
    ) -> None:
        super().__init__(verbose)
        self.window_size = window_size

        self._fidelities:    list = []
        self._successes:     list = []
        self._depths:        list = []

        # public history for post-training plots
        self.fidelity_history:    list = []
        self.success_history:     list = []
        self.depth_history:       list = []
        self.timestep_history:    list = []

    def _on_step(self) -> bool:
        # SB3 puts episode info in self.locals["infos"]
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                # episode just finished — pull fidelity from final info
                fidelity = info.get("fidelity", 0.0)
                success  = info.get("success",  False)
                depth    = info.get("depth",    0)

                self._fidelities.append(fidelity)
                self._successes.append(int(success))
                self._depths.append(depth)

                # keep rolling window
                if len(self._fidelities) > self.window_size:
                    self._fidelities.pop(0)
                    self._successes.pop(0)
                    self._depths.pop(0)

                # log to SB3
                self.logger.record(
                    "train/mean_fidelity", float(np.mean(self._fidelities))
                )
                self.logger.record(
                    "train/success_rate", float(np.mean(self._successes))
                )
                self.logger.record(
                    "train/mean_depth", float(np.mean(self._depths))
                )

                # store for plotting
                self.fidelity_history.append(float(np.mean(self._fidelities)))
                self.success_history.append(float(np.mean(self._successes)))
                self.depth_history.append(float(np.mean(self._depths)))
                self.timestep_history.append(self.num_timesteps)

        return True  # returning False would stop training


class CurriculumCallback(BaseCallback):
    """
    Connects the CurriculumScheduler to the training loop.

    At each episode end it:
      1. Records success/failure with the scheduler
      2. Injects the next target unitary into the environment via reset options
      3. Logs the current difficulty level
    """

    def __init__(
        self,
        curriculum,            # CurriculumScheduler instance
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.curriculum = curriculum

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if done:
                success = info.get("success", False)
                self.curriculum.record_episode(success)

                stats = self.curriculum.get_stats()
                self.logger.record(
                    "curriculum/difficulty",   stats["difficulty"]
                )
                self.logger.record(
                    "curriculum/success_rate", stats["success_rate"]
                )

        return True