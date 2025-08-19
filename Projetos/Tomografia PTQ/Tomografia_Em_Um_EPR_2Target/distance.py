from qutip import *
import numpy as np

def distance(rho_real):
    psi_singlet = (1/np.sqrt(2)) * np.array([0, 1, -1, 0])
    rho_ideal = np.outer(psi_singlet, psi_singlet.conj())

    rho_real_obj = Qobj(rho_real)
    rho_ideal_obj =Qobj(rho_ideal)

    D = tracedist(rho_ideal_obj, rho_real_obj)

    return D