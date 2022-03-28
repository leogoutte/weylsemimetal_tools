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

def WeylHamiltonian(size,kx,kz,tx,ty,tz,g):
    """
    Hamiltonian for Bulk Weyl Semimetal
    Two-node minimal model
    Open in y, closed in x, z
    """
    tr = +1 # -1 for TR, +1 for normal
    # diagonals
    diags_x = np.asarray([tr * tx * np.sin(kx) for _ in range(size)])
    diags_z = np.asarray([tz * (2 + g - np.cos(kx) - np.cos(kz)) for _ in range(size)])

    diag_x = np.kron(np.diag(diags_x),Pauli(1))
    diag_z = np.kron(np.diag(diags_z),Pauli(3))

    diags = diag_x + diag_z

    # hopping
    hop_low = 1j * ty / 2 * np.kron(np.eye(size,k=-1),Pauli(2)) - tz / 2 * np.kron(np.eye(size,k=-1),Pauli(3))
    hop = hop_low + hop_low.conj().T

    MAT = diags + hop

    return MAT

def MetalHamiltonian(size,kx,kz,t,mu):
    """
    Hamiltonian for Bulk Metal 
    Open in y, closed in x, z
    """
    # diagonals
    diags_0 = np.asarray([(-t * (np.cos(kx) + np.cos(kz)) - mu) for _ in range(size)])

    diags = np.kron(np.diag(diags_0),Pauli(0)) 

    # hopping
    hop_low = -t / 2 * np.kron(np.eye(size,k=-1),Pauli(0))
    hop = hop_low + hop_low.conj().T

    MAT = diags + hop

    return MAT

def BulkMetalHamiltonian(kx,ky,kz,t,mu):
    """
    Hamiltonian for the Bulk metal
    """
    H = np.cos(kx)+np.cos(ky)+np.cos(kz)
    H *= -t
    H += -mu
    H_bulk = H * Pauli(0)
    return H_bulk

def TunnellingMatrix(size_n,size_m,r):
    """
    Tunneling matrix for WSM-Metal system
    Returns upper diagonal T^{\dagger}
    """
    Tun_lower = np.zeros((2*size_n,2*size_m),dtype=complex)
    Tun_lower[2*(size_n-1):2*size_n,0:2] = r * Pauli(0)
    
    return Tun_lower

# Full system

def FullHamiltonian(size,kx,kz,t,g,mu,r):
    """
    Hamiltonian for Bulk WSM - Bulk Metal system
    """
    # size of each sample
    new_size = int(size/2) # <- this won't actually add up to size, but ok

    # diagonals
    HWSM = WeylHamiltonian(size=new_size,kx=kx,kz=kz,tx=t,ty=t,tz=t,g=g)
    HMetal = MetalHamiltonian(size=new_size,kx=kx,kz=kz,t=t,mu=mu)
    diags = np.kron((Pauli(0)+Pauli(3))/2,HWSM)+ np.kron((Pauli(0)-Pauli(3))/2,HMetal)

    # tunneling
    Tun_upper = TunnellingMatrix(new_size,new_size,r)
    off_diag = np.kron((Pauli(1)+1j*Pauli(2))/2,Tun_upper) + np.kron((Pauli(1)-1j*Pauli(2))/2,Tun_upper.conj().T) 

    MAT = diags + off_diag

    return MAT

def FullHamiltonianBis(size,kx,kz,t,g,mu,r):
    """
    Hamiltonian for Bulk WSM - Bulk Metal system
    """
    # diagonals
    ky=0 # <- spectral function doesn't change with ky
    HWSM = WeylHamiltonian(size=size,kx=kx,kz=kz,tx=t,ty=t,tz=t,g=g)
    HMetal = BulkMetalHamiltonian(kx=kx,ky=ky,kz=kz,t=t,mu=mu)

    MAT = np.zeros((2*(size+1),2*(size+1)),dtype=complex)

    MAT[0:2*size,0:2*size] = HWSM
    MAT[2*size:2*(size+1),2*size:2*(size+1)] = HMetal

    # tunneling
    Tun_upper = TunnellingMatrix(size,1,r)
    Tun_lower = Tun_upper.conj().T

    MAT[0:2*size,2*size:2*(size+1)] = Tun_upper
    MAT[2*size:2*(size+1),0:2*size] = Tun_lower

    return MAT

def Spectrum(size,kz,t,g,mu,r,bulk=0):
    """
    Energy spectrum for FullHamiltonian
    """
    size_ = size + bulk # for output arrays
    res=100
    ks = np.linspace(-np.pi,np.pi,res)
    Es = np.zeros((2*size_,res),dtype=float)
    Vs = np.zeros((2*size_,2*size_,res),dtype=complex)
    for i in range(res):
        k=ks[i]
        H = FullHamiltonianBis(size=size,kx=k,kz=kz,t=t,g=g,mu=mu,r=r)
        E, V = np.linalg.eigh(H)
        Es[:,i] = E
        Vs[:,:,i] = V
    return ks, Es.T, Vs

def SpectrumZ(size,kx,t,g,mu,r,bulk=0):
    """
    Energy spectrum for FullHamiltonian
    """
    size_ = size + bulk # for output arrays
    res=100
    ks = np.linspace(-np.pi,np.pi,res)
    Es = np.zeros((2*size_,res),dtype=float)
    Vs = np.zeros((2*size_,2*size_,res),dtype=complex)
    for i in range(res):
        k=ks[i]
        H = FullHamiltonianBis(size=size,kx=kx,kz=k,t=t,g=g,mu=mu,r=r)
        E, V = np.linalg.eigh(H)
        Es[:,i] = E
        Vs[:,:,i] = V
    return ks, Es.T, Vs

def FullSpectralFunction(w,size,kx,kz,t,g,mu,r,spin=0):
    """
    Full spectral function calculation
    """
    size_mod=size+1 # new size to account for metal block in G calculation
    # compute Green function
    G = np.linalg.inv(w * np.eye(2 * size_mod) - FullHamiltonianBis(size,kx,kz,t,g,mu,r))

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

    A = - 1 / np.pi * np.imag(np.trace(G[2*(size-1):2*size,2*(size-1):2*size]))

    return A

def FullSpectralFunctionWeylWK(size,res,kx,kz,t=1,g=0,mu=0,m=0.5,r=0.5):
    """
    Return array for plot as a function of energy and momentum
    """
    # set up arrays
    ws = np.linspace(-1.5,1.5,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.03
        A = FullSpectralFunction(w,size,kx,kz,t,g,mu,r)
        As[i] = A

    return As

# basic outline will go as follows:
# find the state we want to investigate
# define spin operators
# compute the expectation values
# plot them in a clever way -- do they depend on momentum? (they could depend on position, but we sum over that with SIGMA = identity * sigma)



def Spin(state,size):
    """
    Computes expectation value of spin along x,y,z axes
    """
    # make sigma operators
    SigmaX = np.kron(np.eye(size+1),Pauli(1))
    SigmaY = np.kron(np.eye(size+1),Pauli(2))
    SigmaZ = np.kron(np.eye(size+1),Pauli(3))
    
    # compute expectation value
    spinX = np.dot(state.conj().T,np.dot(SigmaX,state))
    spinY = np.dot(state.conj().T,np.dot(SigmaY,state))
    spinZ = np.dot(state.conj().T,np.dot(SigmaZ,state))
    
    return spinX, spinY, spinZ