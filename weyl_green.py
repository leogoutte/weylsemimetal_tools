import numpy as np
import scipy.sparse.linalg as ssl

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

# def WeylHamiltonian(size,kx,kz,t,q,w,e):
#     """
#     Hamiltonian for Bulk Weyl Semimetal
#     Open in y, closed in x, z
#     q,w,e extra variables so that i don't have to change code
#     """
#     # diagonals
#     diags_x = np.asarray([t*np.sin(kx) for _ in range(size)])
#     diags_z = np.asarray([t*np.sin(kz) for _ in range(size)])

#     diag_x = np.kron(np.diag(diags_x),Pauli(1)) 
#     diag_z = np.kron(np.diag(diags_z),Pauli(3)) 

#     diags = diag_x + diag_z

#     # hopping
#     hop_low = 1j * t / 2 * np.kron(np.eye(size,k=-1),Pauli(2))
#     hop = hop_low + hop_low.conj().T

#     MAT = diags + hop

#     return MAT

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
    diags_0 = np.asarray([(3/m - mu - 1/m * np.cos(kx) - 1/m * np.cos(kz)) for _ in range(size)])

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
    Tun_lower[2*(size_n-1):2*size_n,0:2,] = r * Pauli(0)
    
    return Tun_lower

def FullHamiltonian(size,kx,kz,t,mu,m,r):
    """
    Hamiltonian for Bulk WSM - Bulk Metal system
    """
    # size of each sample
    new_size = int(size/2) # <- this won't actually add up to size, but ok

    # diagonals
    HWSM = WeylHamiltonian(size=new_size,kx=kx,kz=kz,t=t)
    HMetal = MetalHamiltonian(size=new_size,kx=kx,kz=kz,mu=mu,m=m)
    diags = np.kron(HWSM,(Pauli(0)+Pauli(3)/2)) + np.kron(HMetal,(Pauli(0)-Pauli(3)/2))

    # tunneling
    Tun_lower = TunnellingMatrix(new_size,new_size,r)
    # Tun = Tun_lower + Tun_lower.conj().T
    off_diag = np.kron(Tun_lower,(Pauli(1)+1j*Pauli(2)/2)) + np.kron(Tun_lower.conj().T,(Pauli(1)+1j*Pauli(2)/2)) 

    MAT = diags + off_diag

    return MAT

def GeffWeyl(w,size,kx,kz,t,g,mu,m,r):
    """
    Effective Green function for WSM-Metal system
    integrate over metal states
    m is in units of 1/t, r is in units of t
    """
    new_size_wsm = int(size/2)
    new_size_metal = int(size/2)

    # new_size_wsm = int(size - 2)
    # new_size_metal = 2

    G_inv_weyl = -(w * np.eye(2 * new_size_wsm) - WeylHamiltonian(new_size_wsm,kx,kz,t,t,t,g))

    metal_inverse = - np.linalg.inv(w * np.eye(2 * new_size_metal) - MetalHamiltonian(new_size_metal,kx,kz,mu,m))

    G_inv_metal = TunnellingMatrix(new_size_wsm,new_size_metal,r) @ metal_inverse @ TunnellingMatrix(new_size_wsm,new_size_metal,r).conj().T

    G = np.linalg.inv(G_inv_weyl - G_inv_metal)

    return G

def GeffMetal(w,size,kx,kz,t,g,mu,m,r):
    """
    Effective Green function for WSM-Metal system
    integrate over wsm states
    """
    new_size_metal = int(size/2)
    new_size_wsm = int(size/2)

    # new_size_metal = int(size - 2)
    # new_size_wsm = 2

    G_inv_metal = -(w * np.eye(2 * new_size_metal) - MetalHamiltonian(new_size_metal,kx,kz,mu,m))

    weyl_inverse = np.linalg.inv(w * np.eye(2 * new_size_wsm) - WeylHamiltonian(new_size_wsm,kx,kz,t,t,t,g))

    G_inv_weyl = - TunnellingMatrix(new_size_metal,new_size_wsm,r) @ weyl_inverse @ TunnellingMatrix(new_size_metal,new_size_wsm,r).conj().T

    G = np.linalg.inv(G_inv_metal - G_inv_weyl)

    return G

def SpectralFunctionWeyl(w,size,kx,kz,t,g,mu,m,r):
    """
    Computes the spectral function of a Green function
    A finite delta can be included in w -> w + 1j*0.01
    """
    G = GeffWeyl(w,size,kx,kz,t,g,mu,m,r)

    A = - 1 / np.pi * np.imag(np.trace(G))

    return A

def SpectralFunctionMetal(w,size,kx,kz,t,g,mu,m,r):
    """
    Computes the spectral function of a Green function
    A finite delta can be included in w -> w + 1j*0.01
    """
    G = GeffMetal(w,size,kx,kz,t,g,mu,m,r)

    A = - 1 / np.pi * np.imag(np.trace(G))

    return A

def SpectralFunctionWeylWK(size,kx,kz,t=1,g=0,mu=0,m=0.5,r=0.5):
    """
    Return array for plot as a function of energy and momnetum
    """
    res = size

    # set up arrays
    ws = np.linspace(-1,1,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.01 
        A = SpectralFunctionWeyl(w,size,kx,kz,t,g,mu,m,r)
        As[i] = A

    return As

def SpectralFunctionMetalWK(size,kx,kz,t=1,g=0,mu=0,m=0.5,r=0.5):
    """
    Return array for plot as a function of energy and momnetum
    """
    res = size

    # set up arrays
    ws = np.linspace(-1,1,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.01 
        A = SpectralFunctionMetal(w,size,kx,kz,t,g,mu,m,r)
        As[i] = A

    return As

def SpectralFunctionWeylKK(w,size,kx,t=1,g=0,mu=0,m=0.5,r=0.5):
    """
    Return array for plot as a function of momentum and momnetum
    """
    res = size

    # set up arrays
    kzs = np.linspace(-np.pi,np.pi,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = SpectralFunctionWeyl(w,size,kx,kz,t,g,mu,m,r)
        As[i] = A

    return As

def SpectralFunctionMetalKK(w,size,kx,t=1,g=0,mu=0,m=0.5,r=0.5):
    """
    Return array for plot as a function of momentum and momnetum
    """
    res = size

    # set up arrays
    kzs = np.linspace(-np.pi,np.pi,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = SpectralFunctionMetal(w,size,kx,kz,t,g,mu,m,r)
        As[i] = A

    return As

# function to determine if surface state == True
def Localized(wave,style=0):
    """
    Is the wavefunction localized?
    Equipped to handle array where W[:,i] is ith wave
    """
    
    # make wave into what it was Born to be: probability
    prob = np.abs(wave)**2
    prob_norm = prob / np.sum(prob, axis=0)

    # localization condition: 90% of wave is in 20% of side
    # too strong?
    length = wave.shape[0]
    cut = int(length/5)
    condition = 0.5
    prob_left = np.sum(prob_norm[0:cut,:], axis=0)
    prob_right = np.sum(prob_norm[length-cut:length-1,:], axis=0)

    # make returns
    left = prob_left > condition
    right = prob_right > condition

    # localized on both ends
    if style == 0:
        return np.logical_or(left,right)
    
    # only localized on right side
    elif style == 1:
        return right

    # only localized on left side
    elif style == -1:
        return left


def GSurfaceWeyl(w,size,kx,kz,t,g,mu,m,r):
    """
    Returns the *effective* surface Green function
    Normal surface Green function if r=0
    """
    # effective Green function
    GW = GeffWeyl(w,size,kx,kz,t,g,mu,m,r)

    # diagonalize
    Es , Ws = ssl.eigs(GW, k=int(2*size), return_eigenvectors=True)

    # Localization condition
    localized_bool = Localized(Ws)

    # make G with only localized states
    G = np.diag(Es[localized_bool])

    # print(G)

    return G

def SurfaceSpectralFunctionWeyl(w,size,kx,kz,t,g,mu,m,r):
    """
    Surface spectral function for WSM-Metal system
    """
    G = GSurfaceWeyl(w,size,kx,kz,t,g,mu,m,r)
    A = - 1 / np.pi * np.imag(np.trace(G))
    return A

def SurfaceSpectralFunctionWeylWK(size,kx,kz,t,g,mu,m,r):
    """
    Makes array for Surface spectral function plotted as W vs. K
    """
    res = size

    # set up arrays
    ws = np.linspace(-1,1,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.01 
        A = SurfaceSpectralFunctionWeyl(w,size,kx,kz,t,g,mu,m,r)
        As[i] = A

    return As

def SurfaceSpectralFunctionWeylKK(w,size,kx,t,g,mu,m,r):
    """
    Makes array for Surface spectral function plotted as kx vs. kz
    """
    res = size

    # set up arrays
    kzs = np.linspace(-np.pi,np.pi,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = SurfaceSpectralFunctionWeyl(w,size,kx,kz,t,g,mu,m,r)
        As[i] = A

    return As

def GSurfaceMetal(w,size,kx,kz,t,g,mu,m,r):
    """
    Returns the *effective* surface Green function
    Normal surface Green function if r=0
    """
    # effective Green function
    GW = GeffMetal(w,size,kx,kz,t,g,mu,m,r)

    # diagonalize
    Es , Ws = ssl.eigs(GW, k=int(2*size), return_eigenvectors=True)

    # Localization condition
    localized_bool = Localized(Ws)

    # make G with only localized states
    G = np.diag(Es[localized_bool])

    # print(G)

    return G

def SurfaceSpectralFunctionMetal(w,size,kx,kz,t,g,mu,m,r):
    """
    Surface spectral function for WSM-Metal system
    """
    G = GSurfaceMetal(w,size,kx,kz,t,g,mu,m,r)
    A = - 1 / np.pi * np.imag(np.trace(G))
    return A

def SurfaceSpectralFunctionMetalWK(size,kx,kz,t,g,mu,m,r):
    """
    Makes array for Surface spectral function plotted as W vs. K
    """
    res = size

    # set up arrays
    ws = np.linspace(-1,1,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.01 
        A = SurfaceSpectralFunctionMetal(w,size,kx,kz,t,g,mu,m,r)
        As[i] = A

    return As

def SurfaceSpectralFunctionMetalKK(w,size,kx,t,g,mu,m,r):
    """
    Makes array for Surface spectral function plotted as kx vs. kz
    """
    res = size

    # set up arrays
    kzs = np.linspace(-np.pi,np.pi,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = SurfaceSpectralFunctionMetal(w,size,kx,kz,t,g,mu,m,r)
        As[i] = A

    return As




if __name__ == "__main__":
    import sys

    res = 100
    ks = np.linspace(-np.pi,np.pi,num=res)

    # get kz from argv
    args = sys.argv
    k_idx = int(args[1])
    k = ks[k_idx]

    # run programs
    #AsWeylK = SpectralFunctionWeylKK(w=0.8+1j*0.01,size=100,kx=k,t=1,g=0,mu=0,m=0.5,r=0.5)
    #AsWeylW = SpectralFunctionWeylWK(size=100,kx=k,kz=0,t=1,g=0,mu=0,m=0.5,r=0.5)
    #AsMetalK = SpectralFunctionMetalKK(w=0.8+1j*0.01,size=100,kx=k,t=1,g=0,mu=0,m=0.5,r=0.5)
    #AsMetalW = SpectralFunctionMetalWK(size=100,kx=k,kz=0,t=1,g=0,mu=0,m=0.5,r=0.5)
    
    AsWeylK = SurfaceSpectralFunctionWeylKK(w=0.8+1j*0.01,size=100,kx=k,t=1,g=0,mu=0,m=0.5,r=0.5)
    AsWeylW = SurfaceSpectralFunctionWeylWK(size=100,kx=k,kz=0,t=1,g=0,mu=0,m=0.5,r=0.5)
    AsMetalK = SurfaceSpectralFunctionMetalKK(w=0.8+1j*0.01,size=100,kx=k,t=1,g=0,mu=0,m=0.5,r=0.5)
    AsMetalW = SurfaceSpectralFunctionMetalWK(size=100,kx=k,kz=0,t=1,g=0,mu=0,m=0.5,r=0.5)

    # sum over kz
    # AsWeylK = np.zeros(res,dtype=float)
    # AsWeylW = np.zeros(res,dtype=float)
    # AsMetalK = np.zeros(res,dtype=float)
    # AsMetalW = np.zeros(res,dtype=float)

    # for kz in np.linspace(-np.pi,np.pi,8,endpoint=False):
    #     AsWeylK = AsWeylK + SurfaceSpectralFunctionWeylKK(w=0.8+1j*0.01,size=100,kx=k,t=1,g=0,mu=0,m=0.5,r=0.5)
    #     AsWeylW = AsWeylW + SurfaceSpectralFunctionWeylWK(size=100,kx=k,kz=kz,t=1,g=0,mu=0,m=0.5,r=0.5)
    #     AsMetalK = AsMetalK + SurfaceSpectralFunctionMetalKK(w=0.8+1j*0.01,size=100,kx=k,t=1,g=0,mu=0,m=0.5,r=0.5)
    #     AsMetalW = AsMetalW + SurfaceSpectralFunctionMetalWK(size=100,kx=k,kz=kz,t=1,g=0,mu=0,m=0.5,r=0.5)

    # save text
    np.savetxt("spectral_function_Weyl_KK_{}.csv".format(k_idx), AsWeylK, delimiter = ",")
    np.savetxt("spectral_function_Weyl_WK_{}.csv".format(k_idx), AsWeylW, delimiter = ",")
    np.savetxt("spectral_function_Metal_KK_{}.csv".format(k_idx), AsMetalK, delimiter = ",")
    np.savetxt("spectral_function_Metal_WK_{}.csv".format(k_idx), AsMetalW, delimiter = ",")

### LPBG

    