import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# Lista de matrizes de Pauli em ordem I, X, Y, Z
pauli_list = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]

# Backend para simulação
backend = AerSimulator()

def measurement_circuit2q(i, j, input_circuit, targetA, targetB):
    """
    Prepara um circuito de medição em duas bases (i para A, j para B), com i,j ∈ {'I','X','Y','Z'}.
    - Aplica rotações locais para medir σ_i em targetA e σ_j em targetB.
    - Adiciona um ClassicalRegister(2,'c_tom') e mede: targetA -> c_tom[0], targetB -> c_tom[1].

    Observação (ordem dos bits em get_counts, big-endian do Qiskit):
      - O ÚLTIMO clbit do circuito aparece como bits[0] (menos significativo).
      - Como c_tom é adicionado por último, temos:
           bits[1] ≡ c_tom[0] (resultado de A)
           bits[0] ≡ c_tom[1] (resultado de B)
    """
    circ = input_circuit.copy()

    # Rotação para medir σ_i em A
    if i == 'X':
        circ.h(targetA)
    elif i == 'Y':
        circ.sdg(targetA)
        circ.h(targetA)
    # i == 'Z' ou 'I' -> nada (medida na base computacional)

    # Rotação para medir σ_j em B
    if j == 'X':
        circ.h(targetB)
    elif j == 'Y':
        circ.sdg(targetB)
        circ.h(targetB)
    # j == 'Z' ou 'I' -> nada

    # Bits clássicos de tomografia (sempre adicionados POR ÚLTIMO)
    creg = ClassicalRegister(2, 'c_tom')
    circ.add_register(creg)
    circ.measure(targetA, creg[0])  # A -> c_tom[0] -> bits[1]
    circ.measure(targetB, creg[1])  # B -> c_tom[1] -> bits[0]

    return circ

def tomography2q(my_circuit, targetA, targetB, shots):
    """
    Tomografia didática de dois qubits:
      - Calcula explicitamente os 16 valores ⟨σ_i ⊗ σ_j⟩, i,j ∈ {I,X,Y,Z}.
      - Reconstrói ρ_AB = (1/4) * sum_{i,j} ⟨σ_i⊗σ_j⟩ (σ_i ⊗ σ_j).

    Convenção de extração dos counts (ver measurement_circuit2q):
      bits[1] = resultado de A (c_tom[0]), bits[0] = resultado de B (c_tom[1]).

    Retorna:
      rho (4x4 complex), expectation (dict: (i,j) -> float)
    """
    bases = ['I', 'X', 'Y', 'Z']
    expectation = {}

    for i in bases:
        for j in bases:
            # Normalização: ⟨I⊗I⟩ = 1
            if i == 'I' and j == 'I':
                expectation[(i, j)] = 1.0
                continue

            # Circuito de medida em (i, j)
            circ = measurement_circuit2q(i, j, my_circuit, targetA, targetB)
            tcirc = transpile(circ, backend)
            result = backend.run(tcirc, shots=shots).result()
            counts = result.get_counts()

            # Probabilidades conjuntas (A,B) usando os DOIS últimos bits (c_tom):
            # A ≡ bits[1], B ≡ bits[0]
            p00 = sum(cnt for bits, cnt in counts.items() if bits[1] == '0' and bits[0] == '0')  # A=0,B=0
            p01 = sum(cnt for bits, cnt in counts.items() if bits[1] == '0' and bits[0] == '1')  # A=0,B=1
            p10 = sum(cnt for bits, cnt in counts.items() if bits[1] == '1' and bits[0] == '0')  # A=1,B=0
            p11 = sum(cnt for bits, cnt in counts.items() if bits[1] == '1' and bits[0] == '1')  # A=1,B=1
            total = p00 + p01 + p10 + p11 or 1

            # Marginalização CORRETA quando há 'I'
            if i == 'I' and j != 'I':
                # ⟨I ⊗ σ_j⟩: soma sobre A, pesa apenas B (+1 para B=0, -1 para B=1)
                Eij = (p00 + p10 - p01 - p11) / total
            elif j == 'I' and i != 'I':
                # ⟨σ_i ⊗ I⟩: soma sobre B, pesa apenas A (+1 para A=0, -1 para A=1)
                Eij = (p00 + p01 - p10 - p11) / total
            else:
                # ⟨σ_i ⊗ σ_j⟩: (+1,+1) p00, (+1,-1) p01, (-1,+1) p10, (-1,-1) p11
                Eij = (p00 - p01 - p10 + p11) / total

            expectation[(i, j)] = float(Eij)

    # Reconstrução de ρ_AB
    rho = np.zeros((4, 4), dtype=complex)
    for ii, i in enumerate(bases):
        for jj, j in enumerate(bases):
            rho += expectation[(i, j)] * np.kron(pauli_list[ii], pauli_list[jj])
    rho /= 4.0

    return rho

