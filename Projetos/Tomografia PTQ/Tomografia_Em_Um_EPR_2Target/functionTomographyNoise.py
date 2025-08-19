import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

# --- Criar modelo de ruído ---
noise_model = NoiseModel()

# Erro de despolarização: probabilidade p de o estado virar mistura completamente mista
p1 = 0.01  # erro 1 qubit
p2 = 0.02  # erro 2 qubits

# Erro de relaxação térmica (T1 e T2 em microssegundos, tempo de porta em nanossegundos)
T1 = 50e3
T2 = 70e3
gate_time_1q = 50     # ns
gate_time_2q = 200    # ns

# Definir erros
error_1q = depolarizing_error(p1, 1).compose(
    thermal_relaxation_error(T1, T2, gate_time_1q)
)
error_2q = depolarizing_error(p2, 2).compose(
    thermal_relaxation_error(T1, T2, gate_time_2q).expand(
        thermal_relaxation_error(T1, T2, gate_time_2q)
    )
)

# Associar erros às portas
noise_model.add_all_qubit_quantum_error(error_1q, ["u1", "u2", "u3", "h", "x", "y", "z", "sdg"])
noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "swap"])

# Erro de medida
meas_error = depolarizing_error(0.02, 1)
noise_model.add_all_qubit_quantum_error(meas_error, ["measure"])

# Backend com ruído
backend = AerSimulator(noise_model=noise_model)

# Lista de matrizes de Pauli
pauli_list = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]

def measurement_circuit2q(i, j, input_circuit, targetA, targetB):
    circ = input_circuit.copy()

    if i == 'X':
        circ.h(targetA)
    elif i == 'Y':
        circ.sdg(targetA)
        circ.h(targetA)

    if j == 'X':
        circ.h(targetB)
    elif j == 'Y':
        circ.sdg(targetB)
        circ.h(targetB)

    creg = ClassicalRegister(2, 'c_tom')
    circ.add_register(creg)
    circ.measure(targetA, creg[0])
    circ.measure(targetB, creg[1])
    return circ

def tomography2q_Noises(my_circuit, targetA, targetB, shots):
    bases = ['I', 'X', 'Y', 'Z']
    expectation = {}

    for i in bases:
        for j in bases:
            if i == 'I' and j == 'I':
                expectation[(i, j)] = 1.0
                continue

            circ = measurement_circuit2q(i, j, my_circuit, targetA, targetB)
            tcirc = transpile(circ, backend)
            result = backend.run(tcirc, shots=shots).result()
            counts = result.get_counts()

            p00 = sum(cnt for bits, cnt in counts.items() if bits[1] == '0' and bits[0] == '0')
            p01 = sum(cnt for bits, cnt in counts.items() if bits[1] == '0' and bits[0] == '1')
            p10 = sum(cnt for bits, cnt in counts.items() if bits[1] == '1' and bits[0] == '0')
            p11 = sum(cnt for bits, cnt in counts.items() if bits[1] == '1' and bits[0] == '1')
            total = p00 + p01 + p10 + p11 or 1

            if i == 'I' and j != 'I':
                Eij = (p00 + p10 - p01 - p11) / total
            elif j == 'I' and i != 'I':
                Eij = (p00 + p01 - p10 - p11) / total
            else:
                Eij = (p00 - p01 - p10 + p11) / total

            expectation[(i, j)] = float(Eij)

    rho = np.zeros((4, 4), dtype=complex)
    for ii, i in enumerate(bases):
        for jj, j in enumerate(bases):
            rho += expectation[(i, j)] * np.kron(pauli_list[ii], pauli_list[jj])
    rho /= 4.0
    return rho

