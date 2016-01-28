import xray as xr
from Utilities.plotting import plotMap
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace, convolve
import cartopy.crs as ccrs
from Utilities.distributions import gauss
from Utilities.maputils import makeGrid

projection=ccrs.PlateCarree()


# 3d
# -----
fig, axs = plt.subplots(2, 2, figsize=(15,10),
                       subplot_kw={'projection':projection})

fname='./cfsr.nc'
dset = xr.open_dataset(fname)
tmp = dset.mslp.copy()
tmp[:] = 1
dset['lap'] = laplace(dset.mslp) * tmp
timestep = 2
plotMap(dset.mslp[timestep],ax=axs[0,0]); plt.title("MSLP")
plotMap(dset.lap[timestep],ax=axs[1,0]); plt.title("Laplacian of MSLP")

cLat, cLon, rmax = -16.4, 149.8, 10
mu = 0
sig = 5 * rmax
delta = 3 * rmax
R,theta = makeGrid(cLon, cLat, margin=0, resolution=0.1, minLon=dset.lon.min(),
                maxLon=dset.lon.max(), minLat=dset.lat.min(),
                maxLat=dset.lat.max())
dset['gauss'] = gauss(R,mu,sig) * tmp
plotMap(dset.gauss[timestep],ax=axs[0,1]); plt.title("Guassian")

# 2d laplace
# -----
fig, axs = plt.subplots(2, 2, figsize=(15,10),
                       subplot_kw={'projection':projection})
tmp = dset.mslp[timestep].copy()
tmp[:] = 1
lap = laplace(dset.mslp[timestep]) * tmp
plotMap(dset.mslp[timestep],ax=axs[0,0]); plt.title("MSLP")
plotMap(lap,ax=axs[1,0]); plt.title("Laplacian of MSLP")

weights = gauss(R,mu,sig) * tmp
plotMap(weights,ax=axs[0,1]); plt.title("Guassian")

# 2d gradient
# ------------

fig, axs = plt.subplots(2, 2, figsize=(15,10),
                       subplot_kw={'projection':projection})
gradx, grady = np.gradient(dset.mslp[timestep])
mgrad = np.sqrt(gradx ** 2 + grady ** 2) 
ngrad = mgrad / mgrad.max() * tmp
plotMap(dset.mslp[timestep],ax=axs[0,0]); plt.title("MSLP")
plotMap(ngrad,ax=axs[1,0]); plt.title("Gradient")
plotMap(weights,ax=axs[0,1]); plt.title("Guassian")
plotMap(np.maximum(ngrad,weights),ax=axs[1,1]); plt.title("Max(gradient,gaussian)")

#conv=convolve(dset.gauss[timestep].values[::5,::5], dset.lap[timestep].values[::5,::5])
#plt.imshow(conv)
#dset['con'] =  convolve(dset.gauss[timestep].values, dset.lap[timestep].values) * tmp[0]

plt.show()

