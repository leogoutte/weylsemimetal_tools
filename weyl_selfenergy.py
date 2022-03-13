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
    # diagonals
    diags_x = np.asarray([tx * np.sin(kx) for _ in range(size)])
    diags_z = np.asarray([tz * (2 + g - np.cos(kx) - np.cos(kz)) for _ in range(size)])

    diag_x = np.kron(np.diag(diags_x),Pauli(1)) 
    diag_z = np.kron(np.diag(diags_z),Pauli(3)) 

    diags = diag_x + diag_z

    # hopping
    hop_low = 1j * ty / 2 * np.kron(np.eye(size,k=-1),Pauli(2)) - tz / 2 * np.kron(np.eye(size,k=-1),Pauli(3))
    hop = hop_low + hop_low.conj().T

    MAT = diags + hop

    return MAT

def MetalHamiltonian(size,kx,kz,mu,m):
    """
    Hamiltonian for Bulk Metal 
    Open in y, closed in x, z
    """
    # diagonals
    diags_0 = np.asarray([(2/m - mu - 1/m * np.cos(kx) - 1/m * np.cos(kz)) for _ in range(size)])

    diags = np.kron(np.diag(diags_0),Pauli(0)) 

    # hopping
    hop_low = -1 / (2 * m) * np.kron(np.eye(size,k=-1),Pauli(0))
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