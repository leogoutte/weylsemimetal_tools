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

def BulkModel(kx,ky,kz,tx,ty,tz,g):
    """
    Hamiltonian for the Bulk Weyl Hamiltonian
    """
    M_z = tz * (2 - np.cos(kx) - np.cos(ky) - np.cos(kz) + g) * Pauli(3)
    M_x = tx * np.sin(kx) * Pauli(1)
    M_y = ty * np.sin(ky) * Pauli(2)

    MAT = M_x + M_y + M_z

    return MAT

def States(kz,res,occ=True,tx=1,ty=1,tz=1,g=0):
    """
    Returns a grid of states
    4 dimensions: [kx,ky,kz, 2 band]
    occ is True (occupied band) or False (valence band)
    """
    # 0 if filled, 1 if valence
    if occ:
        index = 0
    else:
        index = 1

    states = np.zeros((res,res,2),dtype=complex)

    for i in range(res):
        kx = -np.pi + i * 2 * np.pi / res 
        for j in range(res):
            ky = -np.pi + j * 2 * np.pi / res 
            _, waves = np.linalg.eigh(BulkModel(kx,ky,kz,tx=tx,ty=ty,tz=tz,g=g))
            states[i,j,:] = waves[:,index]

    return states

def uij(u,v):
    """
    Computes overlap of wavefunctions u, v
    """
    return np.dot(np.conjugate(u),v)

def BerryFlux(n,m,states,res):
    """
    Computes product
    <u_{n,m}|u_{n+1,m}><u_{n+1,m}|u_{n+1,m+1}><u_{n+1,m+1}|u_{n,m+1}><u_{n,m+1}|u_{n,m}>
    Returns the Wilson loop for a given kz
    """
    # for a given kz
    # product over neighbouring sites
    # imposing pbc by virtue of remainder division %
    W = uij(states[n,m,:],states[(n+1)%res,m,:]) 
    W *= uij(states[(n+1)%res,m,:],states[(n+1)%res,(m+1)%res,:])
    W *= uij(states[(n+1)%res,(m+1)%res,:],states[n,(m+1)%res,:])
    W *= uij(states[n,(m+1)%res,:],states[n,m,:])

    return np.arctan2(W.imag,W.real) # might be a minus sign in front

def ChernNumberKz(states,res):
    """
    Discrete sum over all plaquettes (n,m)
    """
    # Chern numbers
    Q = 0

    # Sum over all plaquettes
    for n in range(res):
        for m in range(res):
            Fnm = BerryFlux(n,m,states,res)
            Q += Fnm
    
    Q /= 2 * np.pi

    return np.around(Q,2)

def ChernNumber(res,occ=True,tx=1,ty=1,tz=1,g=0):
    """
    Chern number as a function of kz
    """
    nc = np.zeros(res+1,dtype=float)

    for i in range(res+1):
        kz = -np.pi + i * 2 * np.pi / res 
        st = States(kz,res,occ=occ,tx=tx,ty=ty,tz=tz,g=g)
        nc_kz = ChernNumberKz(st,res)
        nc[i] = nc_kz
    
    return nc

def PhaseDiagram(res,occ=True,tx=1,ty=1,tz=1):
    """
    Chern number as a function of gamma and kz
    Returns 2D array of Chern numbers
    """
    res += 1 # include midpoint
    gs = np.linspace(-2,2,num=res,endpoint=True)
    kzs = np.linspace(-np.pi,np.pi,num=res,endpoint=True)

    CNs = np.zeros((res,res),dtype=int)

    for i in range(res):
        g = gs[i]
        for j in range(res):
            kz = kzs[j]
            st = States(kz,res,occ=occ,tx=tx,ty=ty,tz=tz,g=g)
            C = ChernNumberKz(st,res)
            CNs[i,j] = C

    return CNs