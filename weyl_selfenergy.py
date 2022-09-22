import numpy as np

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

def TunnellingMatrix(size_n,size_m,r):
    """
    Tunneling matrix for WSM-Metal system
    Returns upper diagonal T^{\dagger}
    """
    Tun_lower = np.zeros((2*size_n,2*size_m),dtype=complex)
    Tun_lower[2*(size_n-1):2*size_n,0:2] = r * Pauli(0)
    
    return Tun_lower

def EffectiveHamiltonian(size,w,kx,kz,t,g,mu,r):
    """
    Hamiltonian with a local same-site potential at y=Ly
    """
    # base part
    H = WeylHamiltonian(size=size,kx=kx,kz=kz,tx=t,ty=t,tz=t,g=g)
    # add local potential
    b = w + mu + t * (np.cos(kx) + np.cos(kz))
    Hr = -r**2 / np.sqrt(b**2 - t**2) * Pauli(0)
    H[2*int(size-1):,2*int(size-1):] += Hr

    return H

def HamiltonianEnergies(size,wrange,res,kx,kz,t,g,mu,r):
    """
    Diagonalizes effective hamiltonian for linspace w
    """
    # set up arrays and lists
    ws = np.linspace(-wrange,wrange,num=res)
    Es = np.zeros((res,2*size),dtype=float)

    # loop over energies
    for i in range(res):
        w=ws[i]
        H = EffectiveHamiltonian(size=size,w=w,kx=kx,kz=kz,t=t,g=g,mu=mu,r=r)
        E = np.linalg.eigvalsh(H)
        Es[i,:] = E

    return Es

def SelfConsistentEnergies(size,wrange,res,kx,kz,t,g,mu,r):
    """
    Returns self-consistent energies (the intersection points of E(w) with w)
    Also returns Flag to show if idxs is empty (True if nonempty, False if empty)
    """
    # set up arrays and lists
    ws = np.linspace(-wrange,wrange,num=res)
    idxs = []

    # diagonalize effective hamiltonian
    Es = HamiltonianEnergies(size=size,wrange=wrange,res=res,kx=kx,kz=kz,t=t,g=g,mu=mu,r=r)

    # for each band, find all intersecting points
    for i in range(Es.shape[1]):
        idx = np.argwhere(np.diff(np.sign(Es[:,i]-ws))) # where does the sign of difference flip
        if idx:
            idxs.append(idx[0,0]) # assume only one intersection per band

    if idxs == []:
        return 0, False
    else:
        return list(ws[idxs]), True
    
def SelfConsistentSpectrum(size,wrange,res,kz,t,g,mu,r):
    """
    Spectrum E vs. kz using self-consistent energy method
    """
    # set up arrays
    ws = np.linspace(-wrange,wrange,num=res)
    ks = np.linspace(-1,1,num=res)

    # set up return arrays
    Ks = []
    Es = []

    # find self consistent energies for each kx
    for i in range(res):
        k = ks[i]
        Self_Es, flag = SelfConsistentEnergies(size=size,wrange=wrange,res=res,kx=k,kz=kz,t=t,g=g,mu=mu,r=r)
        if flag == True:
            Es.append(Self_Es)
            Ks.append(list(np.full(len(Self_Es),k)))

    # flatten lists
    flat_ks = [item for sublist in Ks for item in sublist]
    flat_Es = [item for sublist in Es for item in sublist]

    return flat_ks, flat_Es


def SelfEnergyWeyl(w,size,kx,kz,t,g,mu,m,r):
    """
    Self-energy for WSM-Metal system
    integrate over metal states
    m is in units of 1/t, r is in units of t
    """
    # new_size_wsm = int(size/2)
    # new_size_metal = int(size/2)

    new_size_metal = int(size - 1)
    new_size_wsm = 1

    G_inv_weyl = -(w * np.eye(2 * new_size_wsm) - WeylHamiltonian(new_size_wsm,kx,kz,t,t,t,g))

    G_weyl = np.linalg.inv(G_inv_weyl)
    
    G_inv_metal = - (w * np.eye(2 * new_size_metal) - MetalHamiltonian(new_size_metal,kx,kz,mu,m))
    
    G_metal = np.linalg.inv(G_inv_metal)

    SE = TunnellingMatrix(new_size_wsm,new_size_metal,r) @ G_metal @ TunnellingMatrix(new_size_wsm,new_size_metal,r).conj().T
    
    return SE

def SelfEnergyMetal(w,size,kx,kz,t,g,mu,m,r):
    """
    Self-energy for WSM-Metal system
    integrate over WSM states
    m is in units of 1/t, r is in units of t
    """
    # new_size_wsm = int(size/2)
    # new_size_metal = int(size/2)

    new_size_wsm = int(size - 1)
    new_size_metal = 1

    G_inv_weyl = -(w * np.eye(2 * new_size_wsm) - WeylHamiltonian(new_size_wsm,kx,kz,t,t,t,g))

    G_weyl = np.linalg.inv(G_inv_weyl)
    
    G_inv_metal = - (w * np.eye(2 * new_size_metal) - MetalHamiltonian(new_size_metal,kx,kz,mu,m))
    
    G_metal = np.linalg.inv(G_inv_metal)
    
    SE = G_metal @ TunnellingMatrix(new_size_metal,new_size_wsm,r) @ G_weyl @ TunnellingMatrix(new_size_metal,new_size_wsm,r).conj().T @ G_inv_metal
    
    return SE

def SelfEnergyWeylWK(size,res,kx,kz,t,g,mu,m,r):
    """
    Return array for plot as a function of energy and momentum
    """
    # set up arrays
    ws = np.linspace(-1,1,num=res)

    SEs = np.zeros((res),dtype=complex)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.01 
        SE = SelfEnergyWeyl(w,size,kx,kz,t,g,mu,m,r)
        SE_diag = np.linalg.eigvals(SE)
        SEs[i] = SE_diag[0]

    return SEs








if __name__ == "__main__":
    import sys

    res = 100
    ks = np.linspace(-np.pi,np.pi,num=res)

    # get kz from argv
    args = sys.argv
    k_idx = int(args[1])
    k = ks[k_idx]

    # run programs
    SEsWeyl = SelfEnergyWeylWK(size=100,kx=k,kz=-np.pi/2,t=1,g=0,mu=0,m=0.5,r=2.3)

    # save text
    np.savetxt("self_energy_Weyl_WK_{}.csv".format(k_idx), SEsWeyl, delimiter = ",")

### LPBG