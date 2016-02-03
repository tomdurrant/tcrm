import xray as xr
from Utilities.plotting import plotMap
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from Utilities.distributions import gauss
from Utilities.maputils import makeGrid

import unittest
import NumpyTestCase



lap3d=False
lap2d=False

class TestFilters(unittest.TestCase):

    def setUp(self,):
        self.projection=ccrs.PlateCarree()
        fname='./cfsr.nc'
        self.dset = xr.open_dataset(fname)
        fig, self.axs = plt.subplots(2, 2, figsize=(15,10),
                                subplot_kw={'projection':self.projection})
        self.timestep = 2
        mpl=plotMap(self.dset.mslp[self.timestep],ax=self.axs[0,0],robust=True); plt.title("MSLP")
        self.vmin,self.vmax = mpl.get_clim()
        #cLat, cLon, rmax = -16.4, 149.8, 10
        cLat, cLon, rmax = -16.4, 149.8, 30
        mu = 0
        sig = 5 * rmax
        delta = 3 * rmax
        self.R,theta = makeGrid(cLon, cLat, margin=0, resolution=0.1, minLon=self.dset.lon.min(),
                        maxLon=self.dset.lon.max(), minLat=self.dset.lat.min(),
                        maxLat=self.dset.lat.max())
        self.tmp2d = self.dset.mslp[self.timestep].copy()
        plotMap(gauss(self.R,mu,sig) * self.tmp2d,ax=self.axs[0,1]); plt.title("Guassian")
        self.tmp3d = self.dset.mslp.copy()
        self.tmp2d = self.dset.mslp[self.timestep].copy()
        self.tmp2d[:] = 1
        self.tmp3d[:] = 1
        self.gauss = gauss(self.R,mu,sig) * self.tmp2d
        self.bg={'mslp': 100000, 'ugrd10m': 0, 'vgrd10m': 0}

    def test_lap3d(self):
        from scipy.ndimage.filters import laplace
        self.dset['lap'] = laplace(self.dset.mslp) * self.tmp3d
        plotMap(self.dset.lap[self.timestep],ax=self.axs[1,0]); plt.title("Laplacian of MSLP")

    def test_lap2d(self):
        from scipy.ndimage.filters import laplace
        from scipy.ndimage.filters import gaussian_laplace
        from scipy.ndimage.filters import gaussian_filter
        from scipy.ndimage.filters import gaussian_gradient_magnitude
        image = self.dset.mslp[self.timestep].values
        sig=10
        plt.figure()
        plt.imshow(gaussian_filter(image,sig)); plt.title('guassian')
        plt.colorbar()
        plt.figure()
        plt.imshow(gaussian_laplace(image,sig)); plt.title('gl')
        plt.colorbar()
        plt.figure()
        plt.imshow(gaussian_gradient_magnitude(image,sig)); plt.title('glm')
        plt.colorbar()

    def test_gradient_removal(self):
        from scipy.ndimage.filters import gaussian_gradient_magnitude as ggm
        from scipy.ndimage.filters import gaussian_filter as gf
        from scipy.signal import convolve2d
        image = self.dset.mslp[self.timestep].values
        grad = ggm(image,10)
        dist = gf(self.R,10)
        dist =  self.R
        #gntmp = grad/grad.max() * dist.max()/dist
        #gn = grad/grad.max()
        conv = convolve2d(grad/grad.max(),self.gauss/self.gauss.max(), mode='same')
        gn = conv/conv.max()
        plotMap(grad*self.tmp2d,ax=self.axs[1,0],robust=True); plt.title("Gradiant Magnitude of MSLP")
        #out = ((1 - gn) * image) + (gn * self.bg['mslp'])
        plotMap(conv*self.tmp2d,ax=self.axs[1,1]); plt.title("Convolution")
        fig, self.axs = plt.subplots(3, 3, figsize=(18,15),
                                subplot_kw={'projection':self.projection})
        self.plotPair('mslp',gn,0,cmap=plt.get_cmap('viridis'))
        self.plotPair('ugrd10m',gn,1,cmap=plt.get_cmap('RdBu'))
        self.plotPair('vgrd10m',gn,2,cmap=plt.get_cmap('RdBu'))
        plt.show()

    def plotPair(self,var,gn,nn,**kwargs):
        image = self.dset[var][self.timestep].values
        mpl=plotMap(image*self.tmp2d,ax=self.axs[nn,0],robust=True,**kwargs); plt.title(var)
        vmin,vmax = mpl.get_clim()
        out = ((1 - gn) * image) + (gn * self.bg[var])
        mpl=plotMap(out*self.tmp2d,ax=self.axs[nn,1],vmin=vmin,vmax=vmax,**kwargs); plt.title(var+' filtered')
        mpl=plotMap(image - out*self.tmp2d,ax=self.axs[nn,2],robust=True,**kwargs); plt.title(var+' Difference')

    def test_2dgradient(self):
        gradx, grady = np.gradient(self.dset.mslp[self.timestep])
        mgrad = np.sqrt(gradx ** 2 + grady ** 2) 
        ngrad = mgrad / mgrad.max() * self.tmp2d
        plotMap(ngrad,ax=self.axs[1,0]); plt.title("Gradient")
        plotMap(np.maximum(ngrad,self.gauss),ax=self.axs[1,1]); plt.title("Max(gradient,gaussian)")

    def test_skimage_dilate(self):
        from skimage import data, io, filters
        from skimage import img_as_float
        from skimage.morphology import reconstruction
        edges = filters.sobel(self.dset.mslp[self.timestep].values)
        plotMap(edges*self.tmp2d,ax=self.axs[1,0]); plt.title("Skimage")
        image = self.dset.mslp[self.timestep].copy()
        seed = np.copy(image)
        seed[1:-1, 1:-1] = image.min()
        mask = image
        dilated = reconstruction(seed, mask, method='dilation')
        plotMap(image * self.tmp2d,ax=self.axs[1,1])

    def test_watershed(self):
        from skimage.morphology import watershed
        from skimage.feature import peak_local_max
        from scipy import ndimage as ndi
        # Separate the two objects in image 
        # Generate the markers as local maxima of the distance to the
        # background
        # Find peaks of background field within 100km of Pc
        dscale = self.dset.mslp[self.timestep].copy()
        image = dscale.values.copy()
        local_maxi = peak_local_max(-image, indices=False,)
        markers = ndi.label(local_maxi)[0] 
        labels = watershed(image, markers, mask=image)
        plotMap(self.R * self.tmp2d,ax=self.axs[1,0]);plt.title('Distance')
        plotMap(markers * dscale,ax=self.axs[1,1]);plt.title('Separated Peaks')
        plt.show()


if __name__ == "__main__":
    unittest.main()

