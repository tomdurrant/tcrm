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

    def setUp(self, ):
        projection=ccrs.PlateCarree()
        fname='./cfsr.nc'
        self.dset = xr.open_dataset(fname)
        fig, self.axs = plt.subplots(2, 2, figsize=(15,10),
                                subplot_kw={'projection':projection})
        self.timestep = 2
        mpl=plotMap(self.dset.mslp[self.timestep],ax=self.axs[0,0],robust=True); plt.title("MSLP")
        self.vmin,self.vmax = mpl.get_clim()
        cLat, cLon, rmax = -16.4, 149.8, 10
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

    # def test_lap3d(self):
    #     self.dset['lap'] = laplace(self.dset.mslp) * self.tmp3d
    #     plotMap(self.dset.lap[self.timestep],ax=self.axs[1,0]); plt.title("Laplacian of MSLP")

    # def test_lap2d(self):
    #     from scipy.ndimage.filters import laplace
    #     from scipy.ndimage.filters import gaussian_laplace
    #     from scipy.ndimage.filters import gaussian_filter
    #     from scipy.ndimage.filters import gaussian_gradient_magnitude
    #     # lap = gaussian_laplace(self.dset.mslp[self.timestep].values,2) * self.tmp2d
    #     # plotMap(lap,ax=self.axs[1,0],robust=True); plt.title("Laplacian of MSLP")
    #     # lap = gaussian_laplace(self.dset.mslp[self.timestep].values,10) * self.tmp2d
    #     # plotMap(lap,ax=self.axs[1,1],robust=True); plt.title("Laplacian of MSLP")
    #     image = self.dset.mslp[self.timestep].values
    #     sig=10
    #     plt.figure()
    #     plt.imshow(gaussian_filter(image,sig)); plt.title('guassian')
    #     plt.colorbar()
    #     plt.figure()
    #     plt.imshow(gaussian_laplace(image,sig)); plt.title('gl')
    #     plt.colorbar()
    #     plt.figure()
    #     plt.imshow(gaussian_gradient_magnitude(image,sig)); plt.title('glm')
    #     plt.colorbar()

    def test_gradient_removal(self):
        from scipy.ndimage.filters import gaussian_gradient_magnitude as ggm
        from scipy.ndimage.filters import gaussian_filter as gf
        image = self.dset.mslp[self.timestep].values
        grad = ggm(image,10)
        dist = gf(self.R,10)**(1/2.)
        gntmp = grad/grad.max() + dist.max()/dist
        gn = gntmp/gntmp.max()
        plotMap(gn*self.tmp2d,ax=self.axs[1,0],robust=True); plt.title("Gradiant Magnitude of MSLP")
        out = ((1 - gn) * image) + (gn * image.mean())
        plotMap(out*self.tmp2d,ax=self.axs[1,1],vmin=self.vmin, vmax=self.vmax); plt.title("Gradiant Magnitude of MSLP")

    # def test_2dgradient(self):
    #     gradx, grady = np.gradient(self.dset.mslp[self.timestep])
    #     mgrad = np.sqrt(gradx ** 2 + grady ** 2) 
    #     ngrad = mgrad / mgrad.max() * self.tmp2d
    #     plotMap(ngrad,ax=self.axs[1,0]); plt.title("Gradient")
    #     plotMap(np.maximum(ngrad,self.gauss),ax=self.axs[1,1]); plt.title("Max(gradient,gaussian)")

    # def test_skimage_dilate(self):
    #     from skimage import data, io, filters
    #     from skimage import img_as_float
    #     from skimage.morphology import reconstruction
    #     edges = filters.sobel(self.dset.mslp[self.timestep].values)
    #     plotMap(edges*self.tmp2d,ax=self.axs[1,0]); plt.title("Skimage")
    #     image = self.dset.mslp[self.timestep].copy()
    #     seed = np.copy(image)
    #     seed[1:-1, 1:-1] = image.min()
    #     mask = image
    #     dilated = reconstruction(seed, mask, method='dilation')
    #     plotMap(image * self.tmp2d,ax=self.axs[1,1])

    # def test_watershed(self):
    #     from skimage.morphology import watershed
    #     from skimage.feature import peak_local_max
    #     from scipy import ndimage as ndi
    #     # Separate the two objects in image 
    #     # Generate the markers as local maxima of the distance to the
    #     # background
    #     image = self.dset.mslp[self.timestep].values.copy()
    #     #distance = ndi.distance_transform_edt(image)
    #     local_maxi = peak_local_max(image, indices=False,
    #                                 footprint=np.ones((3, 3)), labels=image)
    #     markers = ndi.label(local_maxi)[0] 
    #     labels = watershed(image, markers, mask=image)
    #     #plotMap(-distance * self.tmp2d,ax=self.axs[1,0]);plt.title('Distance')
    #     plotMap(-markers * self.tmp2d,ax=self.axs[1,1]);plt.title('Separated Peaks')


if __name__ == "__main__":
    unittest.main()
    plt.show()

