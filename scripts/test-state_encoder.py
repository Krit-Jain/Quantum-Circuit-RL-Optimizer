import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from quantum_rl.environment.action_space  import QuantumActionSpace
from quantum_rl.environment.state_encoder import StateEncoder

space   = QuantumActionSpace(2)
encoder = StateEncoder(n_qubits=2, max_steps=50, gate_names=space.gate_names)

print(f"obs_dim         : {encoder.obs_dim}")   # expect 79

# encode an identity circuit vs a random target
identity = np.eye(4, dtype=complex)
target   = np.array(
    [[0.5+0.5j, 0.5+0.5j, 0, 0],
     [0.5-0.5j,-0.5+0.5j, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 1]], dtype=complex
)
gate_counts = {name: 0 for name in space.gate_names}

obs = encoder.encode(
    current_unitary = identity,
    target_unitary  = target,
    fidelity        = 0.25,
    circuit_depth   = 5,
    gate_counts     = gate_counts,
)

print(f"obs shape       : {obs.shape}")                 # expect (79,)
print(f"obs dtype       : {obs.dtype}")                 # expect float32
print(f"obs min / max   : {obs.min():.4f} / {obs.max():.4f}")  # within [-1, 1]
print(f"all in [-1,1]   : {bool((obs >= -1).all() and (obs <= 1).all())}")  # True
print(f"fidelity in obs : {obs[64+32+1]:.4f}")          # expect 0.25 (norm_fid slot)