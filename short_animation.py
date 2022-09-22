# the basics
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import seaborn as sns
# sns.set_style("white")

# and our very own
import weyl_green as wg

# set the parameters we'll be using
size = 30
res = 200
wrange = 1.
krange = np.pi
kz = np.pi/2
w = 0

t = 1
g = 0

tm = 0
mu = -4
spin = 0 # both spins
side = 1 #only the interface (for now)

# def SpectralWeights(size,res,wrange,kx,t,g,tm,mu,r,spin,side):
def SpectralWeights(size,res,krange,w,t,g,tm,mu,r,spin,side):
    """
    Function that computes the spectral weights A(w,kx,kz).
    """
    As = np.zeros((res,res),dtype=float)
    kxs = np.linspace(-krange,krange,num=res)
    for i in range(res):
        kx = kxs[i]
        # As[:,i] = wg.FullSpectralFunctionWeylWK(size=size,res=res,wrange=wrange,kx=kx,kz=kz,t=t,g=g,tm=tm,mu=mu,r=r,spin=spin,side=side)
        As[:,i] = wg.FullSpectralFunctionWeylKK(w=w,size=size,res=res,krange=krange,kx=kx,t=t,g=g,tm=tm,mu=mu,r=r,spin=spin,side=side)
    return As

res_r = 100

rs = np.linspace(0,3,num=res_r)

def update(i):
    ax.clear()

    ax.set_title("Spectral weight at the interface of a Weyl semimetal at $\Delta = {:.2}$\n".format(rs[i]), fontsize=40)
    # As = SpectralWeights(size=size,res=res,wrange=wrange,kz=kz,t=t,g=g,tm=tm,mu=mu,r=rs[i],spin=spin,side=side)
    As = SpectralWeights(w=w,size=size,res=res,wrange=wrange,t=t,g=g,tm=tm,mu=mu,r=rs[i],spin=spin,side=side)
    im = ax.imshow(As, cmap='inferno')
    ax.set_ylim(0,res-1)
    ax.set_xlim(0,res-1)
    ax.set_yticks([0,int(res/2)-1/2,res-1])
    ax.set_yticklabels([r"$-\pi$",r"$0$",r"$\pi$"],fontsize=40)
    ax.set_xticks([0,int(res/2)-1/2,res-1])
    ax.set_xticklabels([r"$-\pi$",r"$0$",r"$\pi$"],fontsize=40)
    ax.set_xlabel(r"$k_x$",fontsize=40)
    ax.set_ylabel(r"$E$",fontsize=40)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.5)
    cb = fig.colorbar(im, cax=cax, orientation='vertical')
    cb.set_ticks([])


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(20,20))
    anime = FuncAnimation(
    fig = fig,
    func = update,
    frames = res_r,
    interval = 50)
# anime.save('animation.gif')
anime.save('animation_kk.gif')