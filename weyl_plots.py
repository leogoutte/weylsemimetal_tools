# module for WSM-metal numerics, especially spectral functions

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
    ky=np.pi/2 # <- spectral function doesn't change with ky
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

def FullSpectralFunction(w,size,kx,kz,t,g,mu,r,spin=0):
    """
    Full spectral function calculation
    """
    size_mod=size+1 # new size to account for metal block in G calculation
    # compute Green function
    G = np.linalg.inv(w * np.eye(2 * size) - FullHamiltonian(size,kx,kz,t,g,mu,r))

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

def FullSpectralFunctionWeylWK(size,res,kx,kz,t=1,g=0,mu=0,r=0.5):
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



def GeffWeyl(w,size,kx,kz,t,g,mu,r):
    """
    Effective Green function for WSM-Metal system
    integrate over metal states
    m is in units of 1/t, r is in units of t
    """
    new_size_wsm = int(size/2)
    new_size_metal = int(size/2)

    # new_size_wsm = 2
    # new_size_metal = 28

    G_inv_weyl = (w * np.eye(2 * new_size_wsm) - WeylHamiltonian(new_size_wsm,kx,kz,t,t,t,g))

    metal_inverse = np.linalg.inv(w * np.eye(2 * new_size_metal) - MetalHamiltonian(new_size_metal,kx,kz,t,mu))

    G_inv_metal = TunnellingMatrix(new_size_wsm,new_size_metal,r) @ metal_inverse @ TunnellingMatrix(new_size_wsm,new_size_metal,r).conj().T

    G = np.linalg.inv(G_inv_weyl - G_inv_metal)

    return G

def ZtoX(size):
    """
    Matrix for unitary transformation sigma_z -> sigma_x
    """
    U_pos = np.eye(size)
    U_spin = 1 / np.sqrt(2) * (Pauli(0) - 1j * Pauli(2))
    U = np.kron(U_pos, U_spin)

    return U

def GeffWeylX(w,size,kx,kz,t,g,mu,r):
    """
    Green function rotated by y-axis to sigma_x
    """
    # regular Green function
    G = GeffWeyl(w,size,kx,kz,t,g,mu,r)

    # unitary transformation
    U = ZtoX(int(size/2))

    # rotate
    # print(U.shape)
    # print(G.shape)
    Gx = U @ G @ U.conj().T

    return Gx

def GlobalSpectralFunctionWeyl(w,size,kx,kz,t,g,mu,r,spin=0,spin_dir="z"):
    """
    Computes the spectral function of a Green function
    A finite delta can be included in w -> w + 1j*0.01
    """
    if spin_dir=="z":
        G = GeffWeyl(w,size,kx,kz,t,g,mu,r)
    elif spin_dir=="x":
        G = GeffWeylX(w,size,kx,kz,t,g,mu,r)

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

def SurfaceSpectralFunctionWeyl(w,size,kx,kz,t,g,mu,r,side=0,spin=0,spin_dir="z"):
    """
    Surface spectral function for WSM-Metal system
    """
    if spin_dir=="z":
        G = GeffWeyl(w,size,kx,kz,t,g,mu,r)
    elif spin_dir=="x":
        G = GeffWeylX(w,size,kx,kz,t,g,mu,r)

    edge = int(size/size)

    def G_summ(spin,base,sgn,edge):
        # computes G_sum for given spin and side (base,sgn)
        # both spins
        if spin == 0:
            G_sum = G[base,base]+G[base+sgn*1,base+sgn*1]
            # add remaining edge states
            for i in range(1,edge):
                G_sum += G[base + sgn * (2*i),base + sgn * (2*i)] + G[base + sgn * (2*i+1),base + sgn * (2*i+1)]
        # spin up
        elif spin == +1:
            G_sum = G[base,base]
            # add remaining edge states
            for i in range(1,edge):
                G_sum += G[base + sgn * (2*i),base + sgn * (2*i)]
        # spin down
        elif spin == -1:
            G_sum = G[base+sgn*1,base+sgn*1]
            # add remaining edge states
            for i in range(1,edge):
                G_sum += G[base + sgn * (2*i+1),base + sgn * (2*i+1)]
        return G_sum

    # both sides
    if side == 0:
        # combine both cases
        G_sum = G_summ(spin,0,+1,edge) + G_summ(spin,size-1,-1,edge)

    # left side
    elif side == -1:
        # we start from
        base = 0
        # and we add
        sgn = +1
        # G_sum
        G_sum = G_summ(spin,base,sgn,edge)

    # right side
    elif side == 1:
        # we start from
        base = size-1
        # and we subtract
        sgn = -1
        # G_sum
        G_sum = G_summ(spin,base,sgn,edge)

    A = - 1 / np.pi * np.imag(G_sum)

    return A

def SpectralFunctionWeyl(w,size,kx,kz,t,g,mu,r,side=0,spin=0,surface=False,spin_dir="z"):
    """
    Returns either global or surface spectral function
    """
    if surface:
        A = SurfaceSpectralFunctionWeyl(w=w,size=size,kx=kx,kz=kz,t=t,g=g,mu=mu,r=r,side=side,spin=spin,spin_dir=spin_dir)
    if not surface:
        A = GlobalSpectralFunctionWeyl(w=w,size=size,kx=kx,kz=kz,t=t,g=g,mu=mu,r=r,spin=spin,spin_dir=spin_dir)

    return A

def SpectralFunctionWeylWK(size,res,wrange,kx,kz,t=1,g=0,mu=0,r=0.5,spin=0,side=0,surface=False,spin_dir="z"):
    """
    Return array for plot as a function of energy and momnetum
    """
    # set up arrays
    ws = np.linspace(-wrange,wrange,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.03
        A = SpectralFunctionWeyl(w=w,size=size,kx=kx,kz=kz,t=t,g=g,mu=mu,r=r,side=side,spin=spin,surface=surface,spin_dir=spin_dir)
        As[i] = A

    return As

def SpectralFunctionWeylKK(w,size,res,kzrange,kx,t=1,g=0,mu=0,r=0.5,spin=0,side=0,surface=False,spin_dir="z"):
    """
    Return array for plot as a function of momentum and momnetum
    """
    # fix w
    w += 1j * 0.03

    # set up arrays
    kzs = np.linspace(-kzrange,kzrange,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = SpectralFunctionWeyl(w=w,size=size,kx=kx,kz=kz,t=t,g=g,mu=mu,r=r,side=side,spin=spin,surface=surface,spin_dir=spin_dir)
        As[i] = A

    return As

# big master function

def SpectralFunctionArray(size,res,plot_type="xw",xrange=np.pi,yrange=np.pi,w=0,kx=0,kz=0,t=1,g=0,mu=-2,r=0,surface=False,side=0,spin=0,spin_dir="z"):
    """
    Returns spectral function (as a function of omega, kx, kz, split along spins, on either side) to be plotted
    --------------------Params--------------------
    size, res --> size of system and resolution of plotting, resp.
    plot_type = "xw","zw","xz" --> what spectral function is plotted against
    xrange,yrange --> range of plotting
    w,kx,kz --> physical quantum numbers. only the relevant ones will be used
    t,g,mu,r --> system parameters
    surface = True/False --> use surface spectral function or not
    side --> 0 for both sides, +/- 1 for specific boundary
    spin = 0, +/- 1 --> 0 = all spins, +/- 1 = plit along sigma_i
    spin_dir = "x", "z" --> direction to split spin along
    """
    As = np.zeros((res,res),dtype=float)

    if plot_type == "xw":
        kxs = np.linspace(-xrange,xrange,res)
        for i in range(res):
            kx = kxs[i]
            As[:,i] = SpectralFunctionWeylWK(size=size,res=res,wrange=yrange,kx=kx,kz=kz,t=1,g=0,mu=mu,r=r,spin=spin,side=side,surface=surface,spin_dir=spin_dir)

    elif plot_type == "zw":
        kzs = np.linspace(-xrange,xrange,res)
        for i in range(res):
            kz = kzs[i]
            As[:,i] = SpectralFunctionWeylWK(size=size,res=res,wrange=yrange,kx=kx,kz=kz,t=1,g=0,mu=mu,r=r,spin=spin,side=side,surface=surface,spin_dir=spin_dir) 

    elif plot_type == "xz":
        kxs = np.linspace(-xrange,xrange,res)
        for i in range(res):
            kx= kxs[i]
            As[:,i] = SpectralFunctionWeylKK(w=w,size=size,res=res,kzrange=yrange,kx=kx,t=1,g=0,mu=mu,r=r,spin=spin,side=side,surface=surface,spin_dir=spin_dir) 

    return As









