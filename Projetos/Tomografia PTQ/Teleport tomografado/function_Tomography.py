import numpy as np
from qiskit import *
from qiskit_aer import AerSimulator

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

backend = AerSimulator()

def measurement_circuit(base,input_circuit, target):
    circuit = input_circuit.copy()

    if base == 'X':
        circuit.h(target)
    elif base == 'Y':
        circuit.sdg(target)
        circuit.h(target)

    creg = ClassicalRegister(1, 'c_tom')
    circuit.add_register(creg)
    circuit.measure(target,creg[0])

    return circuit

def tomography(my_circuit, target,shots):
    expectation_values = {}
    all_counts = {}

    for base in ['X','Y','Z']:
        circ       = measurement_circuit(base, my_circuit, target)
        transpiled = transpile(circ, backend)
        result     = backend.run(transpiled, shots=shots).result()
        counts     = result.get_counts()

        # Agora filtramos o primeiro bit da string (bits[0]), que corresponde a c_tom[0]
        p0 = sum(cnt for bits, cnt in counts.items() if bits[0]=='0')
        p1 = sum(cnt for bits, cnt in counts.items() if bits[0]=='1')
        total = p0 + p1 or 1

        expectation_values[base] = (p0 - p1) / total
        all_counts[base] = counts

    rho = 0.5 * (
        pauli_list[0]
        + expectation_values['X'] * pauli_list[1]
        + expectation_values['Y'] * pauli_list[2]
        + expectation_values['Z'] * pauli_list[3]
    )

    return rho, expectation_values

