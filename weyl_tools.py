import numpy as np
import scipy.sparse as ss
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

def RWH_SmallBlock(ky, size, Phi, tx, ty, tz, g):
    """
    Block for the Real Weyl Hamiltonian Matrix 
    Corresponds to a fixed z, varying x
    """
    # set magnetic field to 0
    Phi = 0

    # make diagonals
    cos_diags = [np.cos(2*np.pi*Phi*x + ky) for x in range(size)]
    sin_diags = [np.sin(2*np.pi*Phi*x + ky) for x in range(size)]

    const_diag = tz * (2 + g) * np.kron(np.eye(size),Pauli(3))
    vary_diag = -tz * np.kron(np.diag(cos_diags),Pauli(3)) + ty * np.kron(np.diag(sin_diags),Pauli(2))
    diag = const_diag + vary_diag

    # make off-diagonals x -> x+1
    hop_low = -tz / 2 * np.kron(np.eye(size,k=-1),Pauli(3)) - 1j / 2 * tx * np.kron(np.eye(size,k=-1),Pauli(1))
    hop = hop_low + hop_low.conj().T

    # combine for total matrix
    MAT = diag + hop

    return MAT

def RWH_BigBlock(ky, size, Phi, tx, ty, tz, g):
    """
    Block for the Real Weyl Hamiltonian Matrix
    Corresponds to varying z
    """
    # make diagonals
    blocks = np.zeros((size,2*size,2*size),dtype=complex)

    for i in range(size):
        blocks[i,:,:] = RWH_SmallBlock(ky=ky,size=size,Phi=Phi,tx=tx,ty=ty,tz=tz,g=g)

    diag = ss.block_diag(blocks)

    # make off-diagonals
    hop_low_small = -tz / 2 * np.kron(np.eye(size),Pauli(3))
    hop_low_big = np.kron(np.eye(size,k=-1),hop_low_small)

    hop = hop_low_big + hop_low_big.conj().T

    MAT = diag + hop

    return MAT

def Spectrum(size, res, Phi, tx, ty, tz, g):
    """
    Compute energies of Real Weyl Hamiltonian
    """
    res = 100
    kys = np.linspace(-2*np.pi,2*np.pi,num=res)
    Es = np.zeros((res,size),dtype=float)

    for i in range(res):
        ky = kys[i]
        H = RWH_BigBlock(ky, size, Phi, tx, ty, tz, g)
        E = ssl.eigsh(H, k=size, return_eigenvectors=False)
        Es[i,:] = E

    # make ky array for plotting
    kys_plot = np.full((size,res), kys).T

    return kys_plot, Es

def MetalHamiltonian(size,kx,kz,t,mu):
    """
    Metal Hamiltonian
    """
    # define the identity
    I = np.eye(size)

    # define off-diagonals for hopping x -> x+1
    K = np.eye(size,k=-1)

    # diagonals
    diags = (-2 * t * (np.cos(kx) + np.cos(kz)) - mu) * np.kron(I,Pauli(0))

    # y-hoppping
    yhop = -t * np.kron(K,Pauli(0))
    yhop = yhop + yhop.T.conj()

    MAT = diags + yhop

    return MAT

def MetalSpectrum(size,res,kz,t,mu):
    """
    Compute energies of Real Weyl Hamiltonian
    """
    res = 100
    size_ = 2*size
    ks = np.linspace(-2*np.pi,2*np.pi,num=res)

    Es = np.zeros((res,size_),dtype=float)
    Ks = np.zeros((res,size_),dtype=float)

    for i in range(res):
        k = ks[i]
        H = MetalHamiltonian(size=size,kx=k,kz=kz,t=t,mu=mu)
        E = ssl.eigsh(H, k=size_, return_eigenvectors=False)
        Es[i,:] = E
        Ks[i,:] = np.full(size_,k)

    # make ky array for plotting
    # ks_plot = np.full((size,res), ks).T

    return Ks, Es

def MetalFiniteYSpectrumKx(size,res,kz,t,mu):
    """
    Compute energies of Finite X Hamiltonian
    Plots as a function of kz
    """
    Hdim = int(2*size)
    res = 100
    kxs = np.linspace(-np.pi,np.pi,num=res)
    Es = np.zeros((int(Hdim*res)),dtype=float)
    Localization = np.zeros((int(Hdim*res)),dtype=bool)
    waves = np.zeros((int(Hdim*res), Hdim), dtype=complex)

    for i in range(res):
        kx = kxs[i]
        H = MetalHamiltonian(size,kx,kz,t,mu)
        E, wave = ssl.eigsh(H, k=Hdim, return_eigenvectors=True)
        localized_bool = Localized(wave)
        Es[Hdim*i:Hdim*(i+1)] = E
        waves[Hdim*i:Hdim*(i+1),:] = wave
        Localization[Hdim*i:Hdim*(i+1)] = localized_bool

    # make k array for plotting
    kxs_plot = np.repeat(kxs,Hdim)

    return kxs_plot, Es, waves, Localization
    
def BulkModel(kx,ky,kz,tx,ty,tz,g):
    """
    Hamiltonian for the Bulk Weyl Hamiltonian
    """
    M_z = tz * (2 - np.cos(kx) - np.cos(ky) - np.cos(kz) + g) * Pauli(3)
    M_x = tx * np.sin(kx) * Pauli(1)
    M_y = ty * np.sin(ky) * Pauli(2)

    MAT = M_x + M_y + M_z

    return MAT

def BulkSpectrum(res,kx,ky,tx,ty,tz,g):
    """
    Compute energies of Bulk 2-band Model
    Plots as a function of kz
    """
    kzs = np.linspace(-np.pi,np.pi,num=res)
    Es = np.zeros((int(2*res)),dtype=float)

    for i in range(res):
        kz = kzs[i]
        H = BulkModel(kx,ky,kz,tx,ty,tz,g)
        E = ssl.eigsh(H, k=2, return_eigenvectors=False)
        Es[2*i:2*(i+1)] = E

    # make k array for plotting
    kzs_plot = np.repeat(kzs,2)

    return kzs_plot, Es

def BulkSpectrumSummedOver(res,ky,tx,ty,tz,g):
    """
    Compute energies summed over kx
    to compare with finite length model
    """
    # resolutions are the same in all directions
    resx=res 
    
    kxs = np.linspace(-np.pi,np.pi,num=resx)
    kzs_plot = np.zeros((int(2*res*res)),dtype=float)
    Es = np.zeros((int(2*res*res)),dtype=float)

    for i in range(res):
        kx = kxs[i]
        kzs, E = BulkSpectrum(res=res,kx=kx,ky=ky,tx=tx,ty=ty,tz=tz,g=g)
        kzs_plot[2*resx*i:2*resx*(i+1)] = kzs
        Es[2*resx*i:2*resx*(i+1)] = E

    return kzs_plot, Es

def BulkSpectrumKx(res,ky,kz,tx,ty,tz,g):
    """
    Compute energies of Bulk 2-band Model
    Plots as a function of kx
    """
    kxs = np.linspace(-np.pi,np.pi,num=res)
    Es = np.zeros((int(2*res)),dtype=float)

    for i in range(res):
        kx = kxs[i]
        H = BulkModel(kx,ky,kz,tx,ty,tz,g)
        E = ssl.eigsh(H, k=2, return_eigenvectors=False)
        Es[2*i:2*(i+1)] = E

    # make k array for plotting
    kxs_plot = np.repeat(kxs,2)

    return kxs_plot, Es

def BulkSpectrumSummedOverKx(res,kz,tx,ty,tz,g):
    """
    Compute energies summed over ky
    to compare with finite length model
    """
    # resolutions are the same in all directions
    resy=res 
    
    kys = np.linspace(-np.pi,np.pi,num=resy)
    kxs_plot = np.zeros((int(2*res*res)),dtype=float)
    Es = np.zeros((int(2*res*res)),dtype=float)

    for i in range(res):
        ky = kys[i]
        kxs, E = BulkSpectrumKx(res=res,ky=ky,kz=kz,tx=tx,ty=ty,tz=tz,g=g)
        kxs_plot[2*resy*i:2*resy*(i+1)] = kxs
        Es[2*resy*i:2*resy*(i+1)] = E

    return kxs_plot, Es

def BulkSpectrumKy(res,kx,kz,tx,ty,tz,g):
    """
    Compute energies of Bulk 2-band Model
    Plots as a function of ky
    """
    kys = np.linspace(-np.pi,np.pi,num=res)
    Es = np.zeros((int(2*res)),dtype=float)

    for i in range(res):
        ky = kys[i]
        H = BulkModel(kx,ky,kz,tx,ty,tz,g)
        E = ssl.eigsh(H, k=2, return_eigenvectors=False)
        Es[2*i:2*(i+1)] = E

    # make k array for plotting
    kys_plot = np.repeat(kys,2)

    return kys_plot, Es

def BulkSpectrumSummedOverKy(res,kz,tx,ty,tz,g):
    """
    Compute energies as function of ky summed over kx
    to compare with finite length model
    """
    # resolutions are the same in all directions
    resx=res 
    
    kxs = np.linspace(-np.pi,np.pi,num=resx)
    kys_plot = np.zeros((int(2*res*res)),dtype=float)
    Es = np.zeros((int(2*res*res)),dtype=float)

    for i in range(res):
        kx = kxs[i]
        kys, E = BulkSpectrumKy(res=res,kx=kx,kz=kz,tx=tx,ty=ty,tz=tz,g=g)
        kys_plot[2*resx*i:2*resx*(i+1)] = kys
        Es[2*resx*i:2*resx*(i+1)] = E

    return kys_plot, Es

def FlipGammas(gammas,res,kx,ky,tx,ty,tz):
    """
    Flips through given range of gammas
    """
    # make k and E array
    ks = np.zeros((int(2*res),len(gammas)),dtype=float)
    Es = np.zeros((int(2*res),len(gammas)),dtype=float)

    for i in range(len(gammas)):
        g = gammas[i]
        ks_g, Es_g = BulkSpectrum(res=res,kx=kx,ky=ky,tx=tx,ty=ty,tz=tz,g=g)
        ks[:,i] = ks_g
        Es[:,i] = Es_g

    return ks, Es

def HFiniteX(size, Phi, ky, kz, tx, ty, tz, g):
    """
    Periodic in y, z
    Finite in x 
    For A = Bx in y direction
    i.e. B field in z direction
    Tested for Phi=0 and matches up
    """
    # set magnetic field to 0
    # Phi = 0
    # make sure Phi * size = no. of mag. UC = integer

    # make diagonals
    cos_diags = np.asarray([np.cos(2*np.pi*Phi*x + ky) for x in range(size)])
    sin_diags = np.asarray([np.sin(2*np.pi*Phi*x + ky) for x in range(size)])

    diag_z = tz * np.kron(np.diag(2 + g - cos_diags - np.cos(kz)),Pauli(3)) 
    diag_y = ty * np.kron(np.diag(sin_diags),Pauli(2)) 
    diagonal = diag_y + diag_z

    # make off-diagonals x -> x+1
    hop_low = -tz / 2 * np.kron(np.eye(size,k=-1),Pauli(3)) + 1j / 2 * tx * np.kron(np.eye(size,k=-1),Pauli(1))
    hop = hop_low + hop_low.conj().T

    # combine for total matrix
    MAT = diagonal + hop

    return MAT

def FiniteXSpectrum(size,Phi,res,ky,tx,ty,tz,g):
    """
    Compute energies of Finite X Hamiltonian
    Plots as a function of kz
    """
    Hdim = int(2*size)
    res = 100
    kzs = np.linspace(-np.pi,np.pi,num=res)
    Es = np.zeros((int(Hdim*res)),dtype=float)
    Localization = np.zeros((int(Hdim*res)),dtype=bool)
    waves = np.zeros((int(Hdim*res), Hdim), dtype=complex)

    for i in range(res):
        kz = kzs[i]
        H = HFiniteX(size,Phi=Phi,ky=ky,kz=kz,tx=tx,ty=ty,tz=tz,g=g)
        E, wave = ssl.eigsh(H, k=Hdim, return_eigenvectors=True)
        localized_bool = Localized(wave)
        Es[Hdim*i:Hdim*(i+1)] = E
        waves[Hdim*i:Hdim*(i+1),:] = wave
        Localization[Hdim*i:Hdim*(i+1)] = localized_bool

    # make k array for plotting
    kzs_plot = np.repeat(kzs,Hdim)

    return kzs_plot, Es, waves, Localization

def FiniteXSpectrumKy(size,Phi,res,kz,tx,ty,tz,g):
    """
    Compute energies of Finite X Hamiltonian
    Plots as a function of kz
    """
    Hdim = int(2*size)
    res = 100
    kys = np.linspace(-np.pi,np.pi,num=res)
    Es = np.zeros((int(Hdim*res)),dtype=float)
    Localization = np.zeros((int(Hdim*res)),dtype=bool)
    waves = np.zeros((int(Hdim*res), Hdim), dtype=complex)

    for i in range(res):
        ky = kys[i]
        H = HFiniteX(size,Phi=Phi,ky=ky,kz=kz,tx=tx,ty=ty,tz=tz,g=g)
        E, wave = ssl.eigsh(H, k=Hdim, return_eigenvectors=True)
        localized_bool = Localized(wave)
        Es[Hdim*i:Hdim*(i+1)] = E
        waves[Hdim*i:Hdim*(i+1),:] = wave
        Localization[Hdim*i:Hdim*(i+1)] = localized_bool

    # make k array for plotting
    kys_plot = np.repeat(kys,Hdim)

    return kys_plot, Es, waves, Localization

def Localized(wave):
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

    return np.logical_or(left,right)

    





# def RealWeylHamiltonian