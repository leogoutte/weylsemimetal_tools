import numpy as np
# from scipy.optimize import fsolve

def flip_sign_idxs(array):
    """
    indices of sign flips in array -- equivalent to numeric root finder
    """
    # get the signs of the array
    signs = np.sign(array)
    # difference between all the terms
    difference = np.diff(signs)
    idxs = np.nonzero(difference)[0] # the [0] is needed to get first element of tuple
    return idxs

def SurfaceEnergyRelation(E,branch,p_idx,q_idx,kx,kz,t,g,tm,mu,r):
    """
    Self-consistent energy relation derived from surface theory
    Discrete formalism
    """
    # make variables complex
    if np.size(E) == 1:
        E = complex(E)
    else:
        E = E.astype(complex)

    if np.size(kz) == 1:
        kz = complex(kz)
    else:
        kz = kz.astype(complex)

    if np.size(kx) == 1:
        kx = complex(kx)
    else:
        kx = kx.astype(complex)

    # set branch
    if branch == 0:
        # set it to the correct chiral state
        branch = -np.sign(kx)
    
    # define hamiltonian terms
    g0 = -2 * tm * (np.cos(kx) + np.cos(kz) - 3) - mu 
    g1 = (t * np.sin(kx)).astype(complex)
    g3 = (t * (2 + g - np.cos(kx) - np.cos(kz))).astype(complex)

    # decay parameters
    tmoverp = (g0 - E) / (2) - p_idx * np.sqrt( (g0 - E)**2 / (4) - tm**2) # to make it work when tm = 0

    gamma = (g1**2 + g3**2 + t**2 - E**2) / (2 * g3 * t)
    q = gamma + q_idx * np.sqrt( (gamma)**2 - 1)

    # tunnelling term
    tun = r**2 / (E - g0 + tmoverp)

    # right-hand-side
    rhs = branch * np.sqrt(g1**2 + g3**2 - g3 * t / q)

    # all together (right now (over me))
    ret = (E - tun - rhs) #* (E - tun + rhs)

    return ret

def Energy(branch,p_idx,q_idx,kx,kz,t,g,tm,mu,r):
    """
    Returns the energies given by the self-consistent surface relation for a given set of parameters 
    """
    """
    Finds roots of implicit equation for Energy
    """
    res = 1000
    Espace = np.linspace(-1,1,num=res)
    
    Fspace = SurfaceEnergyRelation(Espace,branch,p_idx,q_idx,kx,kz,t,g,tm,mu,r)
    idxs = flip_sign_idxs(Fspace)
    
    # make the energy the midpoint between sign changes
    if idxs.all() != Espace.shape[0] - 1: 
        Energies = (Espace[idxs] + Espace[idxs+1]) / 2
    else:
        Energies = Espace[idxs]
        
    return Energies

def Energies(branch,p_idx,q_idx,krange,kz,t,g,tm,mu,r,adjusted=False):
    """
    Dispersion wrt kx from self-consistent surface relation
    Adjusted makes sure that the + branch is taken for kx<0 and v.v.
    """
    res = 1000
    kxs = np.linspace(-krange,krange,num=res)
    kxs_ret = np.array([])
    Es = np.array([])
    
    for i in range(res):
        kx = kxs[i]
        # add adjusted condition
        if adjusted:
            if kx <= 0:
                branch = +1
            elif kx > 0:
                branch = -1
        # compute energies
        E = Energy(branch,p_idx,q_idx,kx,kz,t,g,tm,mu,r)
        if E.size > 0:
            Es = np.append(Es,E)
            kxs_ret = np.append(kxs_ret,np.repeat(kx,E.size))
        
    return kxs_ret, Es

def SurfaceEnergyRelationContinuumBis(E,kx,kz,t,g,tm,mu,r):
    """
    Self-consistent energy relation derived from surface theory
    Continuous formalism
    """
    # make variables complex
    if np.size(E) == 1:
        E = complex(E)
    else:
        E = E.astype(complex)

    if np.size(kz) == 1:
        kz = complex(kz)
    else:
        kz = kz.astype(complex)

    if np.size(kx) == 1:
        kx = complex(kx)
    else:
        kx = kx.astype(complex)

    g1 = np.sin(kx)
    hz = 1 + g - np.cos(kx) - np.cos(kz)
    b = E + 2 * tm * (np.cos(kx) + np.cos(kz) - 3) + mu

    # Etilde = E - r**2 / np.sqrt(b**2 - 4 * tm**2)
    Etilde = E - r**2 / (E + mu)

    lp = np.sqrt(2 * (1 + hz) + 2 * np.sqrt(1 + 2 * hz - g1**2 + Etilde**2))
    lm = np.sqrt(2 * (1 + hz) - 2 * np.sqrt(1 + 2 * hz - g1**2 + Etilde**2))

    phi = ((Etilde + hz - lp**2 / 2) / (g1 + lp)) / ((Etilde + hz - lm**2 / 2) / (g1 + lm)) 

    return phi

def SurfaceEnergyRelationContinuum(E,kx,kz,g,mu,r):
    """
    Energy relation derived in continuum limit
    It is assumed t = 1
    """
    # make E complex
    E = E.astype(complex)

    # define key variables
    g1 = np.sin(kx)
    hz = (1 + g - np.cos(kx) - np.cos(kz))
    h0 = -2 * (np.cos(kx) + np.cos(kz) + 1) - mu

    # lambda
    lplus = np.sqrt(2 * (1 + hz) + 2 * np.sqrt(1 + 2 * hz + E**2 - g1**2))
    lminus = np.sqrt(2 * (1 + hz) - 2 * np.sqrt(1 + 2 * hz + E**2 - g1**2))

    # ratio of spins u/v
    wplus = (hz - 1/2 * lplus**2 + E) / (g1 + lplus)
    wminus = (hz - 1/2 * lminus**2 + E) / (g1 + lminus)

    # metal parts
    lmetal = np.sqrt(h0 - E)
    wmetal = (lplus * wminus - lminus * wplus + 2 * (lmetal + r) * (wplus - wminus)) / (lplus - lminus)

    # equation
    eq = (lplus * wplus - (wplus - wmetal) / (wminus - wmetal) * lminus * wminus) + 2 * (lmetal + r) * (wplus - wminus) / (wmetal - wminus)

    return eq

def EnergyContinuum(kx,kz,g,mu,r):
    """
    Finds roots of implicit equation for Energy
    """
    res = 1000
    Espace = np.linspace(-1,1,num=res)
    
    Fspace = SurfaceEnergyRelationContinuum(Espace,kx,kz,g,mu,r)
    idxs = flip_sign_idxs(Fspace)
    
    # make the energy the midpoint between sign changes
    if idxs.all() != Espace.shape[0] - 1: 
        Energies = (Espace[idxs] + Espace[idxs+1]) / 2
    else:
        Energies = Espace[idxs]
        
    return Energies

def EnergiesContinuum(krange,kz,g,mu,r):
    """
    Surface energies plotted for kx
    """
    res = 1000
    kxs = np.linspace(-krange,krange,num=res)
    kxs_ret = np.array([])
    Es = np.array([])
    
    for i in range(res):
        kx = kxs[i]
        E = EnergyContinuum(kx,kz,g,mu,r)
        if E.size > 0:
            Es = np.append(Es,E)
            kxs_ret = np.append(kxs_ret,np.repeat(kx,E.size))
        
    return kxs_ret, Es

# interface subspace

def SelfConsistentEnergyInterface(E,kx,kz,g,M,r):
    """
    Self-consistent equation for energy as a function of kx, kz in interface theory
    """
    # make variables complex
    if np.size(E) == 1:
        E = complex(E)
    else:
        E = E.astype(complex)

    if np.size(kz) == 1:
        kz = complex(kz)
    else:
        kz = kz.astype(complex)

    if np.size(kx) == 1:
        kx = complex(kx)
    else:
        kx = kx.astype(complex)

    g1 = np.sin(kx)
    hz = 1 + g - np.cos(kx) - np.cos(kz)

    lp = np.sqrt(2 * (1 + hz) + 2 * np.sqrt(1 + 2 * hz - g1**2 + (E - r**2 / (E - M))**2))
    lm = np.sqrt(2 * (1 + hz) - 2 * np.sqrt(1 + 2 * hz - g1**2 + (E - r**2 / (E - M))**2))

    phi = ((E + hz - lm**2 / 2 - r**2 / (E - M)) / (g1 + lm)) / ((E + hz - lp**2 / 2 - r**2 / (E - M)) / (g1 + lp)) - 1

    return phi















