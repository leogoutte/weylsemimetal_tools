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

def BulkMetalHamiltonian(kx,ky,kz,t,mu):
    """
    Hamiltonian for Bulk Metal 
    """
    # diagonals
    MAT = (- mu - t * (np.cos(kx) + np.cos(ky) + np.cos(kz))) * Pauli(0)

    return MAT

def BulkKPHamiltonian(size,kx,ky,kz,t,g,mu,r):
    """
    Hamiltonian in k.p approximation
    """
    q=np.pi/size

    # make diagonals
    diags = sl.block_diag(BulkWeylHamiltonian(kx,ky-q,kz,t,t,t,g)/2,
    BulkWeylHamiltonian(kx,ky,kz,t,t,t,g)/2,
    BulkWeylHamiltonian(kx,ky+q,kz,t,t,t,g)/2,
    BulkMetalHamiltonian(kx,ky-q,kz,t,mu)/2,
    BulkMetalHamiltonian(kx,ky,kz,t,mu)/2,
    BulkMetalHamiltonian(kx,ky+q,kz,t,mu)/2)

    # make tunnelling matrix
    tuns = np.zeros((2*3,2*3),dtype=complex)

    tun = r / size * np.exp(-1j * q) * Pauli(0)
    tun0 = r / size * Pauli(0)

    tuns[2*1:2*2,2*0:2*1] = tun
    tuns[2*1:2*2,2*1:2*2] = tun0
    tuns[2*1:2*2,2*2:2*3] = tun.conj().T

    tunn = np.kron(Block("p"),tuns)

    MAT = diags + tunn + tunn.conj().T

    return MAT

def BulkSpectrum(size,res,ky,kz,t,g,mu,r):
    """
    Spectrum for Bulk Hamiltonian with Tunnelling
    """
    s = 12

    kxs = np.linspace(-np.pi,np.pi,num=res)

    Es = np.zeros((res,s),dtype=float)

    for i in range(res):
        kx = kxs[i]
        H = BulkKPHamiltonian(size,kx,ky,kz,t,g,mu,r)
        E = np.linalg.eigvalsh(H)
        Es[i,:] = E

    return kxs, Es

def BulkSpectrumSummedOver(size,res,kz,t,g,mu,r):
    """
    Compute energies summed over ky
    to compare with finite length model
    """
    s = 12

    # resolutions are the same in all directions
    resy=res
    
    kys = np.linspace(-np.pi,np.pi,num=resy)

    Es = np.zeros((res,s*resy),dtype=float)

    for i in range(res):
        ky = kys[i]
        kxs, E = BulkSpectrum(size=size,res=res,ky=ky,kz=kz,t=t,g=g,mu=mu,r=r)
        Es[:,s*i:s*(i+1)] = E

    return kxs, Es



def FullSpectralFunction(w,kx,kz,t,g,mu,r,spin=0):
    """
    Full spectral function calculation
    """
    size = 12
    # compute Green function
    ky=0
    G = np.linalg.inv(w * np.eye(size) - BulkKPHamiltonian(size,kx,ky,kz,t,g,mu,r))

    # both spins summed over
    if spin == 0:
        A = - 1 / np.pi * np.imag(np.trace(G))

    # only up spin
    if spin == 1:
        G_up = np.diag(G)[::2]
        A = - 1 / np.pi * np.imag(np.sum(G_up))

    # only down spin
    if spin == -1:
        G_down = np.diag(G)[1::2]
        A = - 1 / np.pi * np.imag(np.sum(G_down))

    # A = - 1 / np.pi * np.imag(np.trace(G[2*(size-1):2*size,2*(size-1):2*size]))

    return A

def FullSpectralFunctionWeylWK(res,kx,kz,t=1,g=0,mu=0,r=0.5,spin=0):
    """
    Return array for plot as a function of energy and momentum
    """
    # set up arrays
    ws = np.linspace(-1.5,1.5,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.03
        A = FullSpectralFunction(w,kx,kz,t,g,mu,r,spin)
        As[i] = A

    return As
