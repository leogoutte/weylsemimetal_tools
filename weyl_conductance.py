import numpy as np
import scipy.sparse.linalg as ssl

def Pauli(idx):
    """
    Pauli matrices
    """
    if idx==0:
        pmat=np.array(([1,0],[0,1]),dtype=complex)
    elif idx==1:
        pmat=np.array(([0,1],[1,0]),dtype=complex)
    elif idx==2:
        pmat=np.array(([0,-1j],[1j,0]),dtype=complex)
    elif idx==3:
        pmat=np.array(([1,0],[0,-1]),dtype=complex)

    return pmat

def AnalyticalConductance(kx,kz,t,g,mu,r):
    """
    Analytic conductance formula for spin = ->
    kz is z momentum taken wrt weyl point
    """
    g1 = t * np.sin(kx)
    hz = t * (1 + g - np.cos(kx) - np.cos(kz))
    h0 = 2 * t * (np.cos(kx) + np.cos(kz) + 1) + mu + 0 * 1j

    G = r**2 * g1 / ( np.sqrt(g1**2 + hz**2) * np.abs(np.sqrt(h0)))
    
    return np.real(G)


def Kron4(A,B,C,D):
    """
    Nested kronecker products
    First three are for position, last one is for spin
    """
    return np.kron(A,np.kron(B,np.kron(C,D)))

def Kron3(A,B,C):
    """
    Nested kronecker products
    First two are for position, last one is for spin
    """
    return np.kron(A,np.kron(B,C))

def WeylHamiltonian(size,kz,tx,ty,tz,g):
    """
    Open in x, y, closed in z
    np.kron and Kron3 are our friends
    Instead of making blocks and stacking them we can use the kronecker product operator
    """
    # define the identity
    I = np.eye(size)

    # define off-diagonals for hopping x -> x+1
    K = np.eye(size,k=-1)

    # there are 4 kinds of terms:

    # 1. diagonals
    diags = tz * (2 + g - np.cos(kz)) * Kron3(I,I,Pauli(3))

    # 2. x-hoppping
    xhop1 = 1j * tx / 2 * Kron3(K,I,Pauli(1))
    xhop2 = - tz / 2 * Kron3(K,I,Pauli(3))
    xhop = xhop1 + xhop2 + (xhop1+ xhop2).T.conj()

    # 3. y-hoppping
    yhop1 = 1j * ty / 2 * Kron3(I,K,Pauli(2))
    yhop2 = - tz / 2 * Kron3(I,K,Pauli(3))
    yhop = yhop1 + yhop2 + (yhop1+ yhop2).T.conj()

    # 4. z-hoppping
    # zhop1 = - tz / 2 * Kron4(I,I,K,Pauli(3))
    # zhop = zhop1 + zhop1.T.conj()

    MAT = diags + xhop + yhop #+ zhop

    return MAT

def Spectrum(size,res,t,g):
    """
    Spectrum for Hweyl open in x,y
    Helps us understand why we don't see unidimensional conducting states
    """
    Hdim = int(2 * size * size)
    kzs = np.linspace(-np.pi,np.pi,res)

    kzs_plot = np.zeros(int(Hdim * res), dtype=float)
    Es = np.zeros((Hdim * res), dtype=float)

    for i in range(res):
        kz = kzs[i]
        H = WeylHamiltonian(size=size,kz=kz,tx=t,ty=t,tz=t,g=g)
        E = np.linalg.eigvalsh(H)
        # Es[:,i] = E
        kzs_plot[Hdim * i:Hdim * (i + 1)] = np.repeat(kz,Hdim)
        Es[Hdim * i:Hdim * (i + 1)] = E

    return kzs_plot, Es

def Energies(size,kz,t,g):
    return ssl.eigsh(WeylHamiltonian(size,kz,t,t,t,g),k=int(2*size**2),return_eigenvectors=False)

def Conductance(size,E,t,g):
    """
    Trace of transmission matrix defined in LB formalism
    In units of e^2/h
    """
    # 0^+
    eta = 0.0
    # inverse timescale
    rate = 0.1
    # define identity matrices
    I = np.eye(size)
    I_all = Kron3(I,I,Pauli(0))
    # define a generic matrix of zeros to fill
    sample_matrix = np.zeros((size,size),dtype=complex)

    # Weyl hamiltonian
    Hweyl = WeylHamiltonian(size,t,t,t,g)

    # # tunnelling matrix
    # # we will include tunnelling effects via the same-site potential h_r
    # ytun = sample_matrix
    # ytun[0,-1] = r
    # Tunnelling = Kron(I,ytun,I,Pauli(0))

    # GmetInv = (E + 1j * eta) * I - Hmetal

    # self-energies, lead energies, and transmission

    # for delta(x,0)
    top_corner = sample_matrix
    top_corner[0,0] = 1.

    # for delta(x,Lx)
    bottom_corner = sample_matrix
    bottom_corner[-1,-1] = 1.

    # left lead is at x=0 and y=Ly
    left_lead_matrix = Kron3(top_corner,bottom_corner,Pauli(0))

    # right lead is at x=Lx and y=Ly
    right_lead_matrix = Kron3(bottom_corner,bottom_corner,Pauli(0))

    SigmaLret = -1j * rate / 2 * left_lead_matrix
    SigmaRret = -1j * rate / 2 * right_lead_matrix

    GammaL = 1j * (SigmaLret - SigmaLret.conj().T)
    GammaR = 1j * (SigmaRret - SigmaRret.conj().T)

    # Green function

    # inverse retarded Green function
    GretInv = (E + 1j * eta) * I_all - Hweyl - (SigmaLret + SigmaRret) # - Tunnelling @ Gmet @ Tunnelling.conj().T  # not sure about the Sigma terms here
    
    Gret = np.linalg.inv(GretInv)
    Gadv = Gret.conj().T

    # putting everything together
    Transfer = Gret @ GammaL @ Gadv @ GammaR

    return np.real_if_close(np.trace(Transfer))

def ConductanceVEnergy(size,res,t,g):
    Es = np.linspace(-1,1,num=res)
    Ts = np.zeros(res,dtype=complex)
    for i in range(res):
        E = Es[i]
        T = Conductance(size,E,t,g)
        Ts[i] = T
    return Es, Ts




# Script to run on cluster

if __name__ == "__main__":
    size = 15
    res = 20
    t = 1
    g = 0
    # kz = 0

    import sys
    Es = np.linspace(-1,1,num=res)
    # get Es from argv
    args = sys.argv
    idx = int(args[1])
    E = Es[idx]

    T = Conductance(size,E,t,g)

    with open("conductance.csv", "a") as f:
        np.savetxt(f, (E,np.real(T)), delimiter=",")

    # import sys
    # ks = np.linspace(-np.pi,np.pi,num=res)
    # # get Es from argv
    # args = sys.argv
    # idx = int(args[1])
    # k = ks[idx]

    # Es = Energies(size,kz,t,g)
    # np.savetxt("energies_{}.csv".format(idx), Es, delimiter = ",")

### LPBG


