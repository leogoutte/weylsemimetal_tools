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

# def WeylHamiltonian(size,kx,kz,tx,ty,tz,g):
#     """
#     Hamiltonian for Bulk Weyl Semimetal
#     Open in y, closed in x, z
#     q,w,e extra variables so that i don't have to change code
#     """
#     # diagonals
#     diags_x = np.asarray([tx*np.sin(kx) for _ in range(size)])
#     diags_z = np.asarray([tx*np.sin(kz) for _ in range(size)])

#     diag_x = np.kron(np.diag(diags_x),Pauli(1)) 
#     diag_z = np.kron(np.diag(diags_z),Pauli(3)) 

#     diags = diag_x + diag_z

#     # hopping
#     hop_low = 1j * tx / 2 * np.kron(np.eye(size,k=-1),Pauli(2))
#     hop = hop_low + hop_low.conj().T

#     MAT = diags + hop

#     return MAT

def WeylHamiltonian(size,kx,kz,tx,ty,tz,g):
    """
    Hamiltonian for Bulk Weyl Semimetal
    Two-node minimal model
    Open in y, closed in x, z
    """
    tr = +1 # -1 for TR, +1 for normal
    # diagonals
    diags_x = np.asarray([tr * tx * np.sin(kx) for _ in range(size)])
    diags_z = np.asarray([(tz) * (2 + g - np.cos(kx) - np.cos(kz)) for _ in range(size)])

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
    size += bulk
    res=100
    kxs = np.linspace(-np.pi,np.pi,res)
    Es = np.zeros((2*size,res),dtype=float)
    for i in range(res):
        kx=kxs[i]
        H = FullHamiltonian(size,kx,kz,t,g,mu,r)
        E = np.linalg.eigvalsh(H)
        Es[:,i] = E
    return kxs, Es.T

def FullSpectralFunction(w,size,kx,kz,t,g,mu,r,spin=0,side=0):
    """
    Full spectral function calculation
    """
    G = np.linalg.inv(w * np.eye(2 * size) - FullHamiltonian(size,kx,kz,t,g,mu,r))
    edge = int(size/size)

    # def G_summ(spin,base,sgn,edge):
    #     # computes G_sum for given spin and side (base,sgn)
    #     # both spins
    #     if spin == 0:
    #         G_sum = G[base,base]+G[base+sgn*1,base+sgn*1]
    #         # add remaining edge states
    #         for i in range(1,edge):
    #             G_sum += G[base + sgn * (2*i),base + sgn * (2*i)] + G[base + sgn * (2*i+1),base + sgn * (2*i+1)]
    #     # spin up
    #     elif spin == +1:
    #         G_sum = G[base,base]
    #         # add remaining edge states
    #         for i in range(1,edge):
    #             G_sum += G[base + sgn * (2*i),base + sgn * (2*i)]
    #     # spin down
    #     elif spin == -1:
    #         G_sum = G[base+sgn*1,base+sgn*1]
    #         # add remaining edge states
    #         for i in range(1,edge):
    #             G_sum += G[base + sgn * (2*i+1),base + sgn * (2*i+1)]
    #     return G_sum

    # # both sides
    # if side == 0:
    #     # combine both cases
    #     G_sum = G_summ(spin,0,+1,edge) + G_summ(spin,size-1,-1,edge)

    # # left side
    # elif side == -1:
    #     # we start from
    #     base = 0
    #     # and we add
    #     sgn = +1
    #     # G_sum
    #     G_sum = G_summ(spin,base,sgn,edge)

    # # right side
    # elif side == 1:
    #     # we start from
    #     base = size-1
    #     # and we subtract
    #     sgn = -1
    #     # G_sum
    #     G_sum = G_summ(spin,base,sgn,edge)

    # A = - 1 / np.pi * np.imag(np.trace(G[2*(size-1):2*size,2*(size-1):2*size]))
    A = - 1 / np.pi * np.imag(np.trace(G))

    return A

def FullSpectralFunctionWeylWK(size,res,kx,kz,t=1,g=0,mu=0,r=0.5,spin=0,side=0):
    """
    Return array for plot as a function of energy and momentum
    """
    # set up arrays
    ws = np.linspace(-5,5,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.03
        A = FullSpectralFunction(w,size,kx,kz,t,g,mu,r,spin,side)
        As[i] = A

    return As

def FullSpectralFunctionWeylKK(w,size,res,kx,t=1,g=0,mu=0,r=0.5,side=0):
    """
    Return array for plot as a function of momentum and momnetum
    """
    # fix w
    w += 1j * 0.03
    # set up arrays
    kzs = np.linspace(-np.pi,np.pi,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = FullSpectralFunction(w,size,kx,kz,t,g,mu,r,side)
        As[i] = A

    return As   

# Effective system

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

def GeffMetal(w,size,kx,kz,t,g,mu,m,r):
    """
    Effective Green function for WSM-Metal system
    integrate over wsm states
    """
    new_size_metal = int(size/2)
    new_size_wsm = int(size/2)

    # new_size_metal = int(size - 2)
    # new_size_wsm = 2

    G_inv_metal = (w * np.eye(2 * new_size_metal) - MetalHamiltonian(new_size_metal,kx,kz,mu,m))

    weyl_inverse = np.linalg.inv(w * np.eye(2 * new_size_wsm) - WeylHamiltonian(new_size_wsm,kx,kz,t,t,t,g))

    G_inv_weyl = TunnellingMatrix(new_size_metal,new_size_wsm,r) @ weyl_inverse @ TunnellingMatrix(new_size_metal,new_size_wsm,r).conj().T

    G = np.linalg.inv(G_inv_metal - G_inv_weyl)

    return G

# Plotting tools

def SpectralFunctionWeyl(w,size,kx,kz,t,g,mu,r,spin=0):
    """
    Computes the spectral function of a Green function
    A finite delta can be included in w -> w + 1j*0.01
    """
    G = GeffWeyl(w,size,kx,kz,t,g,mu,r)

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

def SpectralFunctionMetal(w,size,kx,kz,t,g,mu,r,spin=0):
    """
    Computes the spectral function of a Green function
    A finite delta can be included in w -> w + 1j*0.01
    """
    G = GeffMetal(w,size,kx,kz,t,g,mu,r)

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

def SpectralFunctionWeylWK(size,res,kx,kz,t=1,g=0,mu=0,r=0.5,spin=0):
    """
    Return array for plot as a function of energy and momnetum
    """
    # set up arrays
    ws = np.linspace(-1.5,1.5,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.03
        A = SpectralFunctionWeyl(w,size,kx,kz,t,g,mu,r,spin)
        As[i] = A

    return As

def SpectralFunctionMetalWK(size,res,kx,kz,t=1,g=0,mu=0,r=0.5,spin=0):
    """
    Return array for plot as a function of energy and momnetum
    """
    # set up arrays
    ws = np.linspace(-1,1,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.01 
        A = SpectralFunctionMetal(w,size,kx,kz,t,g,mu,r,spin)
        As[i] = A

    return As

def SpectralFunctionWeylKK(w,size,res,kx,t=1,g=0,mu=0,r=0.5,spin=0):
    """
    Return array for plot as a function of momentum and momnetum
    """
    # fix w
    w += 1j * 0.03

    # set up arrays
    kzs = np.linspace(-np.pi,np.pi,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = SpectralFunctionWeyl(w,size,kx,kz,t,g,mu,r,spin)
        As[i] = A

    return As

def SpectralFunctionMetalKK(w,size,res,kx,t=1,g=0,mu=0,m=0.5,r=0.5,spin=0):
    """
    Return array for plot as a function of momentum and momnetum
    """
    # fix w
    w += 1j * 0.03

    # set up arrays
    kzs = np.linspace(-np.pi,np.pi,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = SpectralFunctionMetal(w,size,kx,kz,t,g,mu,r,spin)
        As[i] = A

    return As

# Surface system

# function to determine if surface state == True
def Localized(wave,side=0):
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
    if side == 0:
        return np.logical_or(left,right)
    
    # only localized on right side
    elif side == 1:
        return right

    # only localized on left side
    elif side == -1:
        return left

def SurfaceSpectralFunctionWeyl(w,size,kx,kz,t,g,mu,r,side=0,spin=0):
    """
    Surface spectral function for WSM-Metal system
    """
    G = GeffWeyl(w,size,kx,kz,t,g,mu,r)
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

def SurfaceSpectralFunctionMetal(w,size,kx,kz,t,g,mu,m,r,side=0,spin=0):
    """
    Surface spectral function for WSM-Metal system
    """
    G = GeffMetal(w,size,kx,kz,t,g,mu,m,r)
    edge = int(size/10)

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

    # left side
    if side == 0:
        # combine both cases
        G_sum = G_summ(spin,0,+1,edge) + G_summ(spin,size-1,-1,edge)

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

# Plotting tools

def SurfaceSpectralFunctionWeylWK(size,res,kx,kz,t,g,mu,r,side=0,spin=0):
    """
    Makes array for Surface spectral function plotted as W vs. K
    """
    # set up arrays
    ws = np.linspace(-1.5,1.5,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.03 
        A = SurfaceSpectralFunctionWeyl(w,size,kx,kz,t,g,mu,r,side,spin)
        As[i] = A

    return As

def SurfaceSpectralFunctionWeylKK(w,size,res,kx,t,g,mu,m,r,side=0,spin=0):
    """
    Makes array for Surface spectral function plotted as kx vs. kz
    """
    # fix w
    w += 1j * 0.03

    # set up arrays
    kzs = np.linspace(-np.pi,np.pi,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = SurfaceSpectralFunctionWeyl(w,size,kx,kz,t,g,mu,m,r,side,spin)
        As[i] = A

    return As

def SurfaceSpectralFunctionMetalWK(size,res,kx,kz,t,g,mu,m,r,side=0,spin=0):
    """
    Makes array for Surface spectral function plotted as W vs. K
    """
    # set up arrays
    ws = np.linspace(-1,1,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j * 0.05 
        A = SurfaceSpectralFunctionMetal(w,size,kx,kz,t,g,mu,m,r,side,spin)
        As[i] = A

    return As

def SurfaceSpectralFunctionMetalKK(w,size,res,kx,t,g,mu,m,r,side=0,spin=0):
    """
    Makes array for Surface spectral function plotted as kx vs. kz
    """
    # set up arrays
    kzs = np.linspace(-np.pi,np.pi,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(kzs)):
        kz = kzs[i]
        A = SurfaceSpectralFunctionMetal(w,size,kx,kz,t,g,mu,m,r,side,spin)
        As[i] = A

    return As

def ZtoX(size):
    """
    Matrix for unitary transformation sigma_z -> sigma_x
    """
    U_pos = np.eye(size)
    U_spin = 1 / np.sqrt(2) * (Pauli(0) - 1j * Pauli(2))
    U = np.kron(U_pos, U_spin)

    return U

def GeffWeylX(w,size,kx,kz,t,g,mu,m,r):
    """
    Green function rotated by y-axis to sigma_x
    """
    # regular Green function
    G = GeffWeyl(w,size,kx,kz,t,g,mu,m,r)

    # unitary transformation
    U = ZtoX(int(size/2))

    # rotate
    # print(U.shape)
    # print(G.shape)
    Gx = U @ G @ U.conj().T

    return Gx







def AnalyticGreen(size,w,kx,kz,t,g,mu,r):
    """
    Analytic Green function
    """
    tm = 1*t
    mu *= 1
    # define variables
    g1 = t * np.sin(kx)
    g3 = t * (2 + g - np.cos(kx) - np.cos(kz))
    b = w + tm * (np.cos(kx) + np.cos(kz)) + mu

    # diagonals
    diags_const = np.asarray([w for _ in range(size)])
    diags_x = np.asarray([-g1 for _ in range(size)])
    diags_z = np.asarray([-g3 for _ in range(size)])

    diag_const = np.kron(np.diag(diags_const),Pauli(0)) 
    diag_x = np.kron(np.diag(diags_x),Pauli(1)) 
    diag_z = np.kron(np.diag(diags_z),Pauli(3)) 

    diag = diag_const + diag_x + diag_z

    # add in tunnelling term

    diag[2*int(size-1):,2*int(size-1):] += r**2 / np.sqrt(b**2 - tm**2) * Pauli(0)

    hop_m1 = -1j / 2 * t * Pauli(2) + 1 / 2 * t * Pauli(3) # s = 1
    hop_p1 = +1j / 2 * t * Pauli(2) + 1 / 2 * t * Pauli(3) # s = -1
    hop_low =  np.kron(np.eye(size,k=-1),hop_m1)
    hop_up =  np.kron(np.eye(size,k=+1),hop_p1)

    MAT = (diag + hop_low + hop_up)

    # invert G^{-1} and return it:
    return np.linalg.inv(MAT)

def AnalyticSpectralFunctionWeylWK(size,res,kx,kz,t,g,mu,r,side=0):
    """
    Makes array for Surface spectral function plotted as W vs. K
    """
    # set up arrays
    ws = np.linspace(-1.5,1.5,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j*0.03
        G = AnalyticGreen(size,w,kx,kz,t,g,mu,r)
        if side == 1:
            A = - 1 / np.pi * np.imag(np.trace(G[2*(size-1):,2*(size-1):]))
        else:
            A = - 1 / np.pi * np.imag(np.trace(G))
        As[i] = A

    return As


def BulkSpectralFunction(w,kx,ky,kz,r,mu):
    """
    Analytical Bulk spectral function
    """
    g0 = -2 * (np.cos(kx) + np.cos(ky) + np.cos(kz)) - mu
    g1 = np.sin(kx)
    g2 = np.sin(ky)
    g3 = 2 - np.cos(kx) - np.cos(ky) - np.cos(kz)
    ew = np.sqrt(g1**2 + g2**2 + g3**2)

    u2 = 1 / 2 * (Pauli(0) + g1*Pauli(1) + g2*Pauli(2) + g3*Pauli(3))
    v2 = 1 / 2 * (Pauli(0) - g1*Pauli(1) + g2*Pauli(2) + g3*Pauli(3))

    w += 1j * 0.03

    tgmt = r**2 / (w - g0)

    G = u2 * 1 / (w - ew - tgmt) + v2 * 1 / (w + ew - tgmt)

    A = -1 / np.pi * np.real ((G - G.conj().T) / (2 * 1j))

    bra = np.array([1,1]) / np.sqrt(2)

    Ax = bra@ A @ bra.conj().T 

    return Ax

def BulkSpectralFunctionWeylWK(res,kx,ky,kz,r,mu):
    """
    Makes array for bulk spectral function plotted as W vs. K
    """
    # set up arrays
    ws = np.linspace(-1.5,1.5,num=res)

    As = np.zeros((res),dtype=float)

    # loop over them
    for i in range(len(ws)):
        w = ws[i] + 1j*0.03
        A = BulkSpectralFunction(w,kx,ky,kz,r,mu)
        As[i] = A

    return As





# Script to run on cluster

if __name__ == "__main__":
    import sys

    res = 100
    ks = np.linspace(-np.pi,np.pi,num=res)

    # get kz from argv
    args = sys.argv
    k_idx = int(args[1])
    k = ks[k_idx]

    AsFullK = FullSpectralFunctionKK(w=0.8+1j*0.01,size=100,res=res,kx=k,t=1,g=0,mu=0,m=0.5,r=0.5)
    AsFullW = FullSpectralFunctionWK(size=100,res=res,kx=k,kz=0,t=1,g=0,mu=0,m=0.5,r=0.5)
    
    np.savetxt("spectral_function_KK_{}.csv".format(k_idx), AsFullK, delimiter = ",")
    np.savetxt("spectral_function_WK_{}.csv".format(k_idx), AsFullW, delimiter = ",")


### LPBG

    