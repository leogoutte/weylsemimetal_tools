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

def AnalyticalConductance(kx,kz,t,g,tm,mu,r):
    """
    Analytic conductance formula for spin = ->
    kz is z momentum taken wrt weyl point
    """
    # g1 = t * np.sin(kx)
    # hz = t * (1 + g - np.cos(kx) - np.cos(kz))
    # h0 = 2 * t * (np.cos(kx) + np.cos(kz) + 1) + mu + 0 * 1j

    # G = r**2 * g1 / ( np.sqrt(g1**2 + hz**2) * np.abs(np.sqrt(h0)))
    
    # return np.real(G)

    g1 = t * np.sin(kx) + 1j*0
    g3 = t * (2 + g - np.cos(kx) - np.cos(kz)) + 1j*0
    hm = 2 * tm * (3 - np.cos(kx) - np.cos(kz)) - mu + 1j*0

    G = r**2 * g1 * 2 / np.sqrt(4 * g3**2 - (g1**2 + g3**2 + 1)**2) * 1 / np.sqrt(4 * tm**2 - hm**2)

    # G = -r**2 / kx * 1 / np.sqrt(12*tm**2 - 8*tm*mu + mu**2)
    
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

def Kron2(A,B):
    """
    Nested kronecker products
    First is for position, last one is for spin
    """
    return np.kron(A,B)

def WeylHamiltonian(size,kz,kx=0,tx=1,ty=1,tz=1,g=0,open_x=True):
    """
    Open in x, y, closed in z
    If open_x == False, closed in x and kx becomes quantum number
    np.kron and Kron3 are our friends
    Instead of making blocks and stacking them we can use the kronecker product operator
    """
    # define the identity
    I = np.eye(size)

    # define off-diagonals for hopping x -> x+1
    K = np.eye(size,k=-1)

    # there are 4 kinds of terms:

    if open_x:
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

    elif open_x == False:
        # 1. diagonals
        diags = tx * np.sin(kx) * Kron2(I,Pauli(1)) + tz * (2 + g - np.cos(kx) - np.cos(kz)) * Kron2(I,Pauli(3))
    
        # 3. y-hoppping
        yhop1 = 1j * ty / 2 * Kron2(K,Pauli(2))
        yhop2 = - tz / 2 * Kron2(K,Pauli(3))
        yhop = yhop1 + yhop2 + (yhop1+ yhop2).T.conj()
    
        MAT = diags + yhop 

    return MAT

def MetalHamiltonian(size,kz,kx=0,tm=1,mu=-2,open_x=True):
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

    if open_x:

        # 1. diagonals
        diags = (2 * tm * (3 - np.cos(kz)) - mu) * Kron3(I,I,Pauli(0))
    
        # 2. x-hoppping
        xhop1 = - tm * Kron3(K,I,Pauli(0))
        xhop = xhop1 + xhop1.T.conj()
    
        # 3. y-hoppping
        yhop1 = - tm * Kron3(I,K,Pauli(0))
        yhop = yhop1 + yhop1.T.conj()
    
        MAT = diags + xhop + yhop

    elif open_x==False:  

        # 1. diagonals
        diags = (2 * tm * (3 - np.cos(kx) - np.cos(kz)) - mu) * Kron2(I,Pauli(0))
    
        # 2. y-hoppping
        yhop1 = - tm * Kron2(K,Pauli(0))
        yhop = yhop1 + yhop1.T.conj()
    
        MAT = diags + yhop
    
    return MAT

def TunnellingMatrix(size_x,size_n,size_m,r,open_x=True):
    """
    Tunneling matrix for WSM-Metal system
    Returns upper diagonal T^{\dagger}
    size_x is the size of the system in x
    size_n,m are the sizes in the y direction for the WSM and metal, resp.
    """
    if open_x:
        # x direction -- identity
        Tun_lower_x = np.eye(size_x)
        # y direction -- only one nonzero entry
        Tun_lower_y = np.zeros((size_n,size_m),dtype=complex)
        Tun_lower_y[size_n-1,0] = r
        # put it all together
        T_dag = Kron3(Tun_lower_x,Tun_lower_y,Pauli(0))

    elif open_x==False:
        # y direction -- only one nonzero entry
        Tun_lower_y = np.zeros((size_n,size_m),dtype=complex)
        Tun_lower_y[size_n-1,0] = r
        # put it all together
        T_dag = Kron2(Tun_lower_y,Pauli(0))

    return T_dag

# Full system

def FullHamiltonian(size,kz,kx=0,t=1,g=0,tm=1,mu=-2,r=0,open_x=True):
    """
    Hamiltonian for Bulk WSM - Bulk Metal system
    open in x,y, closed in z
    Enveloppes: surface pseudospin, x, y, real spin
    """
    new_size = size
    I = np.eye(size)
    # diagonals
    HWSM = WeylHamiltonian(size=new_size,kz=kz,kx=kx,tx=t,ty=t,tz=t,g=g,open_x=open_x)
    HMetal = MetalHamiltonian(size=new_size,kz=kz,kx=kx,tm=tm,mu=mu,open_x=open_x)

    # put into upper and lower blocks
    diags = np.kron((Pauli(0)+Pauli(3))/2,HWSM)+ np.kron((Pauli(0)-Pauli(3))/2,HMetal)

    # tunneling
    Tun_upper = TunnellingMatrix(size_x=new_size,size_n=new_size,size_m=new_size,r=r,open_x=open_x)
    off_diag = np.kron((Pauli(1)+1j*Pauli(2))/2,Tun_upper) + np.kron((Pauli(1)-1j*Pauli(2))/2,Tun_upper.conj().T) 

    MAT = diags + off_diag

    return MAT

def Spectrum(size,res,kx=0,t=1,g=0,tm=1,mu=-2,r=0,open_x=True):
    """
    Spectrum for H open in x,y
    Helps us understand why we don't see unidimensional conducting states
    """
    if open_x:
        Hdim = int(2 * 2 * size * size)
    elif open_x==False:
        Hdim = int(2 * 2 * size)

    kzs = np.linspace(-np.pi,np.pi,res)

    kzs_plot = np.zeros(int(Hdim * res), dtype=float)
    Es = np.zeros((Hdim * res), dtype=float)

    for i in range(res):
        kz = kzs[i]
        # H = WeylHamiltonian(size=size,kz=kz,tx=t,ty=t,tz=t,g=g)
        H = FullHamiltonian(size=size,kz=kz,kx=kx,t=t,g=g,tm=tm,mu=mu,r=r,open_x=open_x)
        E = np.linalg.eigvalsh(H)
        # Es[:,i] = E
        kzs_plot[Hdim * i:Hdim * (i + 1)] = np.repeat(kz,Hdim)
        Es[Hdim * i:Hdim * (i + 1)] = E

    return kzs_plot, Es

def SpectrumKx(size,res,kz,t,g):
    """
    Spectrum for Hweyl open in y
    Helps us understand why we don't see unidimensional conducting states
    """
    Hdim = int(2 * size)
    kxs = np.linspace(-np.pi,np.pi,res)

    kxs_plot = np.zeros(int(Hdim * res), dtype=float)
    Es = np.zeros((Hdim * res), dtype=float)

    for i in range(res):
        kx = kxs[i]
        H = WeylHamiltonian(size=size,kz=kz,kx=kx,tx=t,ty=t,tz=t,g=g,open_x=False)
        E = np.linalg.eigvalsh(H)
        kxs_plot[Hdim * i:Hdim * (i + 1)] = np.repeat(kx,Hdim)
        Es[Hdim * i:Hdim * (i + 1)] = E

    return kxs_plot, Es

def Energies(size,kz,t,g):
    return ssl.eigsh(WeylHamiltonian(size,kz,t,t,t,g),k=int(2*size**2),return_eigenvectors=False)

def Conductance(size,E,kz,t,g,tm,mu,r,rate):
    """
    Trace of transmission matrix defined in LB formalism
    In units of e^2/h
    """
    # 0^+
    eta = 0.001

    # define identity matrices
    I = np.eye(size)
    I_all = Kron3(I,I,Pauli(0))

    # hamiltonians
    Hweyl = WeylHamiltonian(size=size,kz=kz,tx=t,ty=t,tz=t,g=g)
    Hmetal = MetalHamiltonian(size=size,kz=kz,tm=tm,mu=mu)
    # H = FullHamiltonian(size=size,kz=kz,t=t,g=g,tm=tm,mu=mu,r=r) # this has a size of 2*size^2 x 2*size^2

    # self-energies, lead energies, and transmission

    # for delta(x,0)
    top_corner = np.zeros((size,size),dtype=complex)
    top_corner[0,0] = 1.

    # for delta(x,Lx)
    bottom_corner = np.zeros((size,size),dtype=complex)
    bottom_corner[-1,-1] = 1.

    # left lead is at x=0 and y=Ly
    left_lead_matrix = Kron3(top_corner,I,Pauli(0)) # geometry

    # right lead is at x=Lx and y=Ly
    right_lead_matrix = Kron3(bottom_corner,I,Pauli(0))

    SigmaLret = -1j * rate / 2 * left_lead_matrix
    SigmaRret = -1j * rate / 2 * right_lead_matrix

    GammaL = 1j * (SigmaLret - SigmaLret.conj().T)
    GammaR = 1j * (SigmaRret - SigmaRret.conj().T)

    # Green function

    # metal green function (don't include Sigmas in this one)
    GmetInv = (E + 1j * eta) * I_all - Hmetal
    Gmet = np.linalg.inv(GmetInv) 

    # tunnelling term
    T_dag = TunnellingMatrix(size_x=size,size_n=size,size_m=size,r=r)
    TUN = T_dag @ Gmet @ T_dag.conj().T

    # inverse retarded Green function
    GretInv = (E + 1j * eta) * I_all - Hweyl - TUN - (SigmaLret + SigmaRret)
    
    Gret = np.linalg.inv(GretInv)
    Gadv = Gret.conj().T

    # putting everything together
    Transfer = Gret @ GammaL @ Gadv @ GammaR

    return np.real_if_close(np.trace(Transfer))

def ConductanceFull(size,E,kz,t,g,tm,mu,r,rate):
    """
    Trace of transmission matrix defined in LB formalism
    In units of e^2/h
    """
    # define full system size
    size_full = int(2*size)

    # 0^+
    eta = 0.001

    # define identity matrices
    I = np.eye(size)
    I_full = np.eye(size_full)
    I_top = np.kron(np.array([[1,0],[0,0]]),I) # for y-dependencies in full Hamiltonian
    I_all = Kron3(I,I_full,Pauli(0))

    # hamiltonians
    # Hweyl = WeylHamiltonian(size=size,kz=kz,tx=t,ty=t,tz=t,g=g)
    # Hmetal = MetalHamiltonian(size=size,kz=kz,tm=tm,mu=mu)
    H = FullHamiltonian(size=size,kz=kz,t=t,g=g,tm=tm,mu=mu,r=r) # this has a size of 2*size^2 x 2*size^2

    # self-energies, lead energies, and transmission

    # for delta(x,0)
    top_corner = np.zeros((size,size),dtype=complex)
    top_corner[0,0] = 1.

    # for delta(x,Lx)
    bottom_corner = np.zeros((size,size),dtype=complex)
    bottom_corner[-1,-1] = 1.

    # left lead is at x=0 and y=Ly
    left_lead_matrix = Kron3(top_corner,I_full,Pauli(0)) # geometry

    # right lead is at x=Lx and y=Ly
    right_lead_matrix = Kron3(bottom_corner,I_full,Pauli(0))

    SigmaLret = -1j * rate / 2 * left_lead_matrix
    SigmaRret = -1j * rate / 2 * right_lead_matrix

    GammaL = 1j * (SigmaLret - SigmaLret.conj().T)
    GammaR = 1j * (SigmaRret - SigmaRret.conj().T)

    # Green function

    # metal green function (don't include Sigmas in this one)
    # GmetInv = (E + 1j * eta) * I_all - Hmetal
    # Gmet = np.linalg.inv(GmetInv) 

    # tunnelling term
    # T_dag = TunnellingMatrix(size_x=size,size_n=size,size_m=size,r=r)
    # TUN = T_dag @ Gmet @ T_dag.conj().T

    # inverse retarded Green function
    # print(I_all.shape)
    # print(H.shape)
    # print(SigmaLret.shape)
    GretInv = (E + 1j * eta) * I_all - H - (SigmaLret + SigmaRret)
    
    Gret = np.linalg.inv(GretInv)
    Gadv = Gret.conj().T

    # putting everything together
    Transfer = Gret @ GammaL @ Gadv @ GammaR

    return np.real_if_close(np.trace(Transfer))

def ConductanceVEnergy(size,res,Erange,kz,t,g,tm,mu,r,rate):
    Es = np.linspace(-Erange,Erange,num=res)
    Ts = np.zeros(res,dtype=float)
    for i in range(res):
        E = Es[i]
        T = Conductance(size,E,kz,t,g,tm,mu,r,rate)
        Ts[i] = T
    return Es, Ts

def SumConductance(size,res_es,t,g,tm,mu,r,rate):
    # sum over kz
    res_k = 50
        
    kzs = np.linspace(-np.pi,np.pi,num=res_k)
    
    Conductance = np.zeros((res_k,res_es),dtype=float)

    for i in range(res_k):
        kz = kzs[i]
        Es, Ts = ConductanceVEnergy(size=size,res=res_es,Erange=1.5,kz=kz,t=t,g=g,tm=tm,mu=mu,r=r,rate=rate)
        Conductance[i,:] = Ts

    return Conductance

def ConductanceClosed(size,E,kz,kx,t,g,tm,mu,r,rate):
    """
    Computes conductance across the barrier 
    Leads are placed at each end of the sample.
    """
    # 0^+
    eta = 0.001
    # define identity matrices
    new_size = int(size/2)
    I = np.eye(size)
    I_all = Kron2(I,Pauli(0))

    # hamiltonians
    # Hweyl = WeylHamiltonian(size=size,kz=kz,kx=kx,tx=t,ty=t,tz=t,g=g,open_x=False)
    # Hmetal = MetalHamiltonian(size=size,kz=kz,kx=kx,tm=tm,mu=mu,open_x=False)
    H = FullHamiltonian(size=new_size,kz=kz,kx=kx,t=t,g=g,tm=tm,mu=mu,r=r,open_x=False) # <- H is of size 'size'

    # self-energies, lead energies, and transmission

    # for delta(x,0)
    top_corner = np.zeros((size,size),dtype=complex)
    top_corner[0,0] = 1.

    # for delta(x,Lx)
    bottom_corner = np.zeros((size,size),dtype=complex)
    bottom_corner[-1,-1] = 1.

    # matrix_for_y = np.zeros((size,size),dtype=complex)
    # matrix_for_y[-1,-1] = 1
    # matrix_for_y[-2,-2] = 1
    # matrix_for_y[-3,-3] = 1

    # left lead is at x=0 and y=Ly
    left_lead_matrix = Kron2(top_corner,Pauli(0)) # geometry

    # right lead is at x=Lx and y=Ly
    right_lead_matrix = Kron2(bottom_corner,Pauli(0))

    SigmaLret = -1j * rate / 2 * left_lead_matrix
    SigmaRret = -1j * rate / 2 * right_lead_matrix

    GammaL = 1j * (SigmaLret - SigmaLret.conj().T)
    GammaR = 1j * (SigmaRret - SigmaRret.conj().T)

    # Green function

    # inverse retarded Green function
    GretInv = (E + 1j * eta) * I_all - H  - (SigmaLret + SigmaRret)
    
    Gret = np.linalg.inv(GretInv)
    Gadv = Gret.conj().T

    # putting everything together
    Transfer = Gret @ GammaL @ Gadv @ GammaR

    return np.real_if_close(np.trace(Transfer))

def ConductanceClosedVEnergy(size,res,Erange=1.5,kz=0,kx=0,t=1,g=0,tm=1,mu=0,r=0,rate=1):
    Es = np.linspace(-Erange,Erange,num=res)
    Ts = np.zeros(res,dtype=float)
    for i in range(res):
        E = Es[i]
        T = ConductanceClosed(size=size,E=E,kz=kz,kx=kx,t=t,g=g,tm=tm,mu=mu,r=r,rate=rate)
        Ts[i] = T
    return Es, Ts



# Script to run on cluster

if __name__ == "__main__":
    size = 40
    res = 50
    r = 2.3
    kz = 0
    t = 1
    g = 0
    tm = 0
    mu = -4
    Erange = 1.5
    rate = 1

    Es0, Ts0 = ConductanceVEnergy(size=size,res=res,Erange=Erange,kz=kz,t=t,g=g,tm=tm,mu=mu,r=0,rate=rate)
    Es, Ts = ConductanceVEnergy(size=size,res=res,Erange=Erange,kz=kz,t=t,g=g,tm=tm,mu=mu,r=r,rate=rate)
    
    # save data
    np.savetxt("conductance_data_0_r_{}.csv".format(r), (Es0,Ts0), delimiter=",")
    np.savetxt("conductance_data_r_{}.csv".format(r), (Es,Ts), delimiter=",")


    # G0 = SumConductance(size=20,res_es=100,t=1,g=0,tm=0,mu=-4,r=0,rate=1)
    # G23 = SumConductance(size=20,res_es=100,t=1,g=0,tm=0,mu=-4,r=2.3,rate=1)

    # np.savetxt("conductance_sum_0.csv", G0, delimiter=",")
    # np.savetxt("conductance_sum_23.csv", G23, delimiter=",")


### LPBG

