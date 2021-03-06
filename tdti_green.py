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

def BulkBHZHamiltonian(kx,ky,A,M,p):
    """
    Hamiltonian for the Bulk BHZ system
    """
    M_z = M * (1 + p - np.cos(kx) - np.cos(ky)) * Pauli(3)
    M_x = A * np.sin(kx) * Pauli(1)
    M_y = A * np.sin(ky) * Pauli(2)

    MAT = M_x + M_y + M_z

    return MAT

def BHZHamiltonian(size,kx,A,M,p):
    """
    Hamiltonian for Bulk Weyl Semimetal
    Two-node minimal model
    Open in y, closed in x, z
    """
    # diagonals
    diags_x = np.asarray([A * np.sin(kx) for _ in range(size)])
    diags_z = np.asarray([M * (1 + p - np.cos(kx)) for _ in range(size)])

    diag_x = np.kron(np.diag(diags_x),Pauli(1)) 
    diag_z = np.kron(np.diag(diags_z),Pauli(3)) 

    diags = diag_x + diag_z

    # hopping
    hop_low = 1j * A / 2 * np.kron(np.eye(size,k=-1),Pauli(2)) - M / 2 * np.kron(np.eye(size,k=-1),Pauli(3))
    hop = hop_low + hop_low.conj().T

    MAT = diags + hop

    return MAT

def BHZSpectrum(size,res,A=5,M=2.5,p=3):
    kxs = np.linspace(-np.pi,np.pi,res)
    Es = np.zeros((2*size,res),dtype=float)
    for i in range(res):
        kx=kxs[i]
        H = BHZHamiltonian(size,kx,A,M,p)
        E = np.linalg.eigvalsh(H)
        Es[:,i] = E
    return kxs, Es.T

def MetalHamiltonian2D(size,kx,mu,m):
    """
    Hamiltonian for Bulk Metal 
    Open in y, closed in x, z
    """
    # diagonals
    diags_0 = np.asarray([(2/m - mu - 1/m * np.cos(kx)) for _ in range(size)])

    diags = np.kron(np.diag(diags_0),Pauli(0)) 

    # hopping
    hop_low = -1 / (2 * m) * np.kron(np.eye(size,k=-1),Pauli(0))
    hop = hop_low + hop_low.conj().T

    MAT = diags + hop

    return MAT

def MetalSpectrum(size,res,mu=0,m=0.5):
    kxs = np.linspace(-np.pi,np.pi,res)
    Es = np.zeros((2*size,res),dtype=float)
    for i in range(res):
        kx=kxs[i]
        H = MetalHamiltonian2D(size,kx,mu,m)
        E = np.linalg.eigvalsh(H)
        Es[:,i] = E
    return kxs, Es.T

def TunnellingMatrix(size_n,size_m,r):
    """
    Tunneling matrix for WSM-Metal system
    Returns upper diagonal T^{\dagger}
    """
    Tun_lower = np.zeros((2*size_n,2*size_m),dtype=complex)
    Tun_lower[2*(size_n-1):2*size_n,0:2] = r * Pauli(0)
    
    return Tun_lower

def FullHamiltonian(size,kx,A,M,p,mu,m,r):
    """
    Hamiltonian for 2DTI - Metal system
    """
    # size of each sample
    new_size = int(size/2) # <- this won't actually add up to size, but ok

    # diagonals
    H2DTI = BHZHamiltonian(size=new_size,kx=kx,A=A,M=M,p=p)
    HMetal = MetalHamiltonian2D(size=new_size,kx=kx,mu=mu,m=m)
    diags = np.kron((Pauli(0)+Pauli(3))/2,H2DTI)+ np.kron((Pauli(0)-Pauli(3))/2,HMetal)

    # tunneling
    Tun_upper = TunnellingMatrix(new_size,new_size,r)
    off_diag = np.kron((Pauli(1)+1j*Pauli(2))/2,Tun_upper) + np.kron((Pauli(1)-1j*Pauli(2))/2,Tun_upper.conj().T) 

    MAT = diags + off_diag

    return MAT

def FullHamiltonian2DTI(size,kx,A,M,p,mu,m,r):
    """
    Hamiltonian for 2DTI - 2DTI system
    """
    # size of each sample
    new_size = int(size/2) # <- this won't actually add up to size, but ok

    # diagonals
    H2DTIp = BHZHamiltonian(size=new_size,kx=kx,A=A,M=M,p=p)
    H2DTIm = BHZHamiltonian(size=new_size,kx=kx,A=A,M=-M,p=p)
    diags = np.kron((Pauli(0)+Pauli(3))/2,H2DTIp)+ np.kron((Pauli(0)-Pauli(3))/2,H2DTIm)

    # tunneling
    Tun_upper = TunnellingMatrix(new_size,new_size,r)
    off_diag = np.kron((Pauli(1)+1j*Pauli(2))/2,Tun_upper) + np.kron((Pauli(1)-1j*Pauli(2))/2,Tun_upper.conj().T) 

    MAT = diags + off_diag

    return MAT

def Spectrum(size,A,M,p,mu,m,r):
    """
    Energy spectrum for FullHamiltonian
    """
    res=100
    s=2
    kxs = np.linspace(-np.pi,np.pi,res)
    Es = np.zeros((s*size,res),dtype=float)
    for i in range(res):
        kx=kxs[i]
        H = FullHamiltonian(size,kx,A,M,p,mu,m,r)
        E = np.linalg.eigvalsh(H)
        Es[:,i] = E
    return kxs, Es.T

def FullSpectralFunction(w,size,kx,A,M,p,mu,m,r,spin=0):
    """
    Full spectral function calculation
    """
    # compute Green function
    G = np.linalg.inv(w * np.eye(2 * size) - BHZHamiltonian(size,kx,A,M,p))

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

    return A

def FullSpectralFunctionWK(size,res,kx,A,M,p,mu,m,r,spin=0):
    """
    Return array for plot as a function of energy and momentum
    """
    # set up arrays
    ws = np.linspace(-A,A,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.5
        A = FullSpectralFunction(w,size,kx,A,M,p,mu,m,r,spin)
        As[i] = A

    return As

def BulkKPHamiltonian(kx,ky,A,M,p,r):
    """
    Hamiltonian with size 4
    """
    # make diagonals
    diags = sl.block_diag(BulkBHZHamiltonian(kx,ky,A,M,p),
    BulkBHZHamiltonian(kx,ky,A,M,p),
    BulkBHZHamiltonian(kx,ky,A,M,p),
    BulkBHZHamiltonian(kx,ky,A,M,p),
    BulkBHZHamiltonian(kx,ky,A,-M,p),
    BulkBHZHamiltonian(kx,ky,A,-M,p),
    BulkBHZHamiltonian(kx,ky,A,-M,p),
    BulkBHZHamiltonian(kx,ky,A,-M,p))

    # make tunnelling matrix
    tuns = np.zeros((2*4,2*4))

    tun = r / 4 * Pauli(0)

    tuns[2*2:2*3,2*0:2*1] = -1 * tun
    tuns[2*2:2*3,2*1:2*2] = -1j * tun
    tuns[2*2:2*3,2*2:2*3] = 1 * tun
    tuns[2*2:2*3,2*3:2*4] = 1j * tun

    tunn = np.kron(Block("p"),tuns)

    MAT = diags + tunn + tunn.conj().T

    return MAT

def BulkSpectrum(res,ky,A,M,p,r):
    """
    Spectrum for Bulk Hamiltonian with Tunnelling
    """
    s = 16

    kxs = np.linspace(-np.pi,np.pi,num=res)

    Es = np.zeros((res,s),dtype=float)

    for i in range(res):
        kx = kxs[i]
        H = BulkKPHamiltonian(kx,ky,A,M,p,r)
        E = np.linalg.eigvalsh(H)
        Es[i,:] = E

    return kxs, Es

def BulkSpectrumSummedOver(res,A,M,p,r):
    """
    Compute energies summed over ky
    to compare with finite length model
    """
    s = 16

    # resolutions are the same in all directions
    resy=res
    
    kys = np.linspace(-np.pi,np.pi,num=resy)

    Es = np.zeros((res,s*resy),dtype=float)

    for i in range(res):
        ky = kys[i]
        kxs, E = BulkSpectrum(res=res,ky=ky,A=A,M=M,p=p,r=r)
        Es[:,s*i:s*(i+1)] = E

    return kxs, Es












