import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from quantum_rl.training.curriculum import CurriculumScheduler

print("=" * 50)
print("  Training Infrastructure — sanity checks")
print("=" * 50)

# ── Test 1: curriculum difficulty ────────────────────
sched = CurriculumScheduler(n_qubits=2, initial_difficulty=0.1, seed=42)

print(f"\n[1] Initial state")
print(f"  difficulty : {sched.difficulty}")       # 0.1
print(f"  stats      : {sched.get_stats()}")

# sample a target
U = sched.sample_target()
print(f"  target shape   : {U.shape}")            # (4, 4)
print(f"  is unitary     : {np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10)}")  # True

# ── Test 2: auto promotion ───────────────────────────
print(f"\n[2] Simulate 20 successes → should promote")
for _ in range(20):
    sched.record_episode(success=True)
print(f"  difficulty after 20 wins : {sched.difficulty}")   # > 0.1

# ── Test 3: demotion ─────────────────────────────────
print(f"\n[3] Simulate 20 failures → should demote")
for _ in range(20):
    sched.record_episode(success=False)
print(f"  difficulty after 20 fails: {sched.difficulty}")   # back down

# ── Test 4: difficulty 0 → near identity ────────────
print(f"\n[4] Difficulty=0 → U should be identity")
U0 = sched._make_unitary(0.0)
print(f"  ||U - I||_F : {np.linalg.norm(U0 - np.eye(4)):.6f}")  # ~0.0

# ── Test 5: difficulty 1 → unitary check ────────────
print(f"\n[5] Difficulty=1 → still a valid unitary")
U1 = sched._make_unitary(1.0)
err = np.linalg.norm(U1 @ U1.conj().T - np.eye(4))
print(f"  ||U U† - I||_F : {err:.2e}")  # < 1e-10

print("\n  All checks done.")