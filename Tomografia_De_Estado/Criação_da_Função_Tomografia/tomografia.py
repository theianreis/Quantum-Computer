import numpy as np
from qiskit import *
from qiskit_aer import AerSimulator
import scipy.linalg as la
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.states.densitymatrix import DensityMatrix

pauli_list = [
    np.eye(2),
    np.array([[0.0, 1.0], [1.0, 0.0]]),
    np.array([[0, -1.0j], [1.0j, 0.0]]),
    np.array([[1.0, 0.0], [0.0, -1.0]]),
]

bases = {
    'I': pauli_list[0],
    'X': pauli_list[1],
    'Y': pauli_list[2],
    'Z': pauli_list[3],
}

backend=AerSimulator()

# Aplica transformação de base + medida
def measurement_circuit(base, input_circuit, target):
    circuit = input_circuit.copy()
    
    if base == 'X':
        circuit.h(target)
    elif base == 'Y':
        circuit.sdg(target)
        circuit.h(target)
    
    creg = ClassicalRegister(1, 'c')
    circuit.add_register(creg)
    circuit.measure(target, creg[0])

    return circuit

# Função principal de tomografia
def tomography(my_circuit, target, shots):
    expectation_values = {}
    all_counts = {}

    for base in ['X', 'Y', 'Z']:
        circ = measurement_circuit(base, my_circuit, target)
        transpiled = transpile(circ, backend)
        job = backend.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()

        p0 = counts.get('0', 0)
        p1 = counts.get('1', 0)
        total = p0 + p1 if p0 + p1 > 0 else 1
        expectation_values[base] = (p0 - p1) / total
        all_counts[base] = counts

    rho = 0.5 * (
        pauli_list[0]
        + expectation_values['X'] * pauli_list[1]
        + expectation_values['Y'] * pauli_list[2]
        + expectation_values['Z'] * pauli_list[3]
    )

    return rho, expectation_values, result