# run as: python -c "..."
import sys
sys.path.insert(0, "src")

from quantum_rl.environment.action_space import QuantumActionSpace

a2 = QuantumActionSpace(2)
a3 = QuantumActionSpace(3)

print(f"2-qubit actions : {a2.n_actions}")   # expect 40
print(f"3-qubit actions : {a3.n_actions}")   # expect 66
print(f"Gate types      : {a2.gate_names}")  # expect 12 types
print(f"Action 0 label  : {a2.get_action_label(0)}")   # expect H_q0
print(f"Matrix shape    : {a2.get_gate_matrix(0).shape}")  # expect (4, 4)