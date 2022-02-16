import numpy as np
import scipy.sparse.linalg as ssl
import scipy.linalg as sl

def Pauli(idx):
    """
    Pauli matrices
    """
    if idx==0:
        pmat=np.identity(2, dtype=float)
    elif idx==1:
        pmat=np.array(([0,1],[1,0]),dtype=float)
    elif idx==2:
        pmat=np.array(([0,-1j],[1j,0]),dtype=complex)
    elif idx==3:
        pmat=np.array(([1,0],[0,-1]),dtype=float)
    
    return pmat

def Block(idx):
    """
    Upper (u), Lower (l), Plus (p) and Minus (m) blocks for np.kron
    """
    if idx=="u":
        pmat=(Pauli(0) + Pauli(3)) / 2
    elif idx=="l":
        pmat=(Pauli(0) - Pauli(3)) / 2
    elif idx=="p":
        pmat=(Pauli(1) + 1j*Pauli(2)) / 2
    elif idx=="m":
        pmat=(Pauli(1) - 1j*Pauli(2)) / 2
    
    return pmat

def BulkWeylHamiltonian(kx,ky,kz,tx,ty,tz,g):
    """
    Hamiltonian for the Bulk Weyl Hamiltonian
    """
    M_z = tz * (2 - np.cos(kx) - np.cos(ky) - np.cos(kz) + g) * Pauli(3)
    M_x = tx * np.sin(kx) * Pauli(1)
    M_y = ty * np.sin(ky) * Pauli(2)

    MAT = M_x + M_y + M_z

    return MAT

def BulkMetalHamiltonian(kx,ky,kz,mu,m):
    """
    Hamiltonian for Bulk Metal 
    """
    # diagonals
    MAT = (3/m - mu - 1/m * (np.cos(kx) + np.cos(ky) + np.cos(kz))) * Pauli(0)

    return MAT

def BulkKPHamiltonian(kx,ky,kz,t,g,mu,m,r):
    """
    Hamiltonian in k.p approximation
    """
    q=np.pi

    diags = sl.block_diag(BulkWeylHamiltonian(kx,ky,kz,t,t,t,g),
    BulkWeylHamiltonian(kx,ky+q,kz,t,t,t,g),
    BulkMetalHamiltonian(kx,ky,kz,mu,m),
    BulkMetalHamiltonian(kx,ky+q,kz,mu,m))

    tun = r * np.exp(-1j * q) * Pauli(0)
    tunn = np.kron(Block("m"),tun) + np.kron(Block("p"),tun.conj().T)
    tuns = np.kron(Block("m"),tunn) + np.kron(Block("p"),tunn.conj().T)

    MAT = diags + tuns

    return MAT

def BulkSpectrum(ky,kz,t,g,mu,m,r):
    """
    Spectrum for Bulk Hamiltonian with Tunnelling
    """
    res = 100
    s = 8

    kxs = np.linspace(-np.pi,np.pi,num=res)

    Es = np.zeros((s,res),dtype=float)
    # Kxs = np.zeros((s*res),dtype=float)

    for i in range(res):
        kx = kxs[i]
        H = BulkKPHamiltonian(kx,ky,kz,t,g,mu,m,r)
        E = np.linalg.eigvalsh(H)
        Es[:,i] = E
        # Kxs[s*i:s*(i+1)] = np.repeat(kx,s)

    return kxs, Es.T
