from qiskit import *
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator
import numpy as np

def DensityMatrix(my_circuit):
    # Estado puro
    psi = Statevector.from_instruction(my_circuit)

    # Valores esperados
    expX = np.real(psi.expectation_value(Pauli("X")))
    expY = np.real(psi.expectation_value(Pauli("Y")))
    expZ = np.real(psi.expectation_value(Pauli("Z")))
    exp_values = [expX,expY,expZ]

    # Matriz densidade pela fórmula
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # Matriz densidade 
    rho = 0.5 * (I + expX*X + expY*Y + expZ*Z)

    return rho,exp_values