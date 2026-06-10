import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from quantum_rl.environment.quantum_env import QuantumCircuitEnv

print("=" * 50)
print("  QuantumCircuitEnv — sanity checks")
print("=" * 50)

# ── Test 1: init ──────────────────────────────────────
env2 = QuantumCircuitEnv(n_qubits=2)
env3 = QuantumCircuitEnv(n_qubits=3)
print(f"\n[1] Init")
print(f"  2-qubit obs_dim  : {env2.observation_space.shape[0]}")   # 79
print(f"  3-qubit obs_dim  : {env3.observation_space.shape[0]}")   # 271
print(f"  2-qubit n_actions: {env2.action_space.n}")               # 40

# ── Test 2: reset ─────────────────────────────────────
obs, info = env2.reset(seed=42)
print(f"\n[2] Reset")
print(f"  obs shape        : {obs.shape}")       # (79,)
print(f"  obs dtype        : {obs.dtype}")       # float32
print(f"  all in [-1,1]    : {bool((obs>=-1).all() and (obs<=1).all())}")  # True
print(f"  initial fidelity : {info['fidelity']:.6f}")  # small random value

# ── Test 3: step ──────────────────────────────────────
obs2, reward, terminated, truncated, info2 = env2.step(0)
print(f"\n[3] Step (action=0, gate=H_q0)")
print(f"  obs shape        : {obs2.shape}")      # (79,)
print(f"  reward           : {reward:.4f}")
print(f"  terminated       : {terminated}")      # False (unlikely on step 1)
print(f"  truncated        : {truncated}")       # False
print(f"  fidelity         : {info2['fidelity']:.6f}")
print(f"  depth            : {info2['depth']}")  # 1

# ── Test 4: full random episode ───────────────────────
env2.reset(seed=0)
total_reward = 0.0
for step in range(50):
    action = env2.action_space.sample()
    _, r, terminated, truncated, info3 = env2.step(action)
    total_reward += r
    if terminated or truncated:
        break

print(f"\n[4] Random episode")
print(f"  steps run        : {info3['depth']}")
print(f"  total reward     : {total_reward:.4f}")
print(f"  final fidelity   : {info3['fidelity']:.6f}")
print(f"  success          : {info3['success']}")

# ── Test 5: identity target (trivial case) ────────────
print(f"\n[5] Identity target (F should start at 1.0)")
identity = np.eye(4, dtype=complex)
obs_id, info_id = env2.reset(options={"target_unitary": identity})
print(f"  fidelity at reset: {info_id['fidelity']:.6f}")  # 1.0

# ── Test 6: all reward types ──────────────────────────
print(f"\n[6] Reward types")
for rt in ["binary", "dense", "log", "hybrid"]:
    e = QuantumCircuitEnv(n_qubits=2, reward_type=rt, seed=1)
    e.reset()
    _, r, _, _, _ = e.step(0)
    print(f"  {rt:8s} reward step 1: {r:.4f}")

print("\n  All checks done.")