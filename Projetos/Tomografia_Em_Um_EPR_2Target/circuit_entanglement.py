from qiskit import *
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

def entanglement(n=0):
    # 1) Cria registradores quânticos Alice e Bob
    alice = QuantumRegister(1, 'Alice')
    bob   = QuantumRegister(1, 'Bob')
    regs_q = [alice, bob]

    # 2) Se n>0, adiciona registrador extra "ext" com n qubits
    if n > 0:
        ext = QuantumRegister(n, 'ext')
        regs_q.append(ext)

    # 3) Cria registrador clássico com 1 bit
    c = ClassicalRegister(2, 'c')

    # 4) Monta o circuito incluindo TODOS os registradores
    qc = QuantumCircuit(*regs_q, c, name=f'entanglement(n={n})')

    # Cria o estado singlete: (|01> - |10>)/sqrt(2)
    qc.x(bob[0])          # Bob em |1>
    qc.h(alice[0])        # Alice em (|0> + |1>)/sqrt(2)
    qc.cx(alice[0], bob[0])
    qc.z(bob[0])          # Inverte fase de |10>
    # 6) Se n>0, aplica cadeia de swaps: Bob ↔ ext[0] ↔ ext[1] ↔ …
    if n > 0:
        qc.swap(bob[0], ext[0])
        for i in range(1, n):
            qc.swap(ext[i-1], ext[i])

    return qc