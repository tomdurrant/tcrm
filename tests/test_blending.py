import os
import sys
import unittest
import cPickle
import NumpyTestCase
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from wind.windmodels import *

try:
    import pathLocate
except:
    from tests import pathLocate

# Add parent folder to python path
unittest_dir = pathLocate.getUnitTestDirectory()
sys.path.append(pathLocate.getRootDirectory())

def plotMap(windfield,lat,lon, ax,ptype='pcolormesh',cblabel=None,**kwargs):
    if ptype=='pcolormesh':
        mpl = ax.pcolormesh(lon, lat, windfield,
                            transform=ccrs.PlateCarree(),**kwargs)
    elif ptype=='contour':
        mpl = ax.contour(lon, lat, windfield,
                         transform=ccrs.PlateCarree(),zorder=2, **kwargs)
        mpl = ax.contourf(lon, lat, windfield,
                          transform=ccrs.PlateCarree(),zorder=1, **kwargs)
    else:
        raise Exception("Plot type %s not recognised" % ptype)
    coast = NaturalEarthFeature(category='physical', scale='50m',
                                facecolor='gray',
                                edgecolor='black',name='coastline')
    ax.add_feature(coast)
    plt.axes(ax)
    cb = plt.colorbar(mpl, fraction=0.046, pad=0.04, orientation='vertical')
    if cblabel:
        cb.set_label(cblabel)
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.xformatter = LONGITUDE_FORMATTER
    return ax


class TestWindField(NumpyTestCase.NumpyTestCase):

    def setUp(self):
        pkl_file = open(os.path.join(unittest_dir, 'test_data', 'windFieldTestData.pck'), 'r')
        self.R = cPickle.load(pkl_file)
        self.lam = cPickle.load(pkl_file)
        self.rMax = cPickle.load(pkl_file)
        self.f = cPickle.load(pkl_file)
        self.V = cPickle.load(pkl_file)
        self.Z = cPickle.load(pkl_file)
        self.vFm = cPickle.load(pkl_file)
        self.thetaFm = cPickle.load(pkl_file)
        self.thetaMax = cPickle.load(pkl_file)
        self.test_kepert_Ux = cPickle.load(pkl_file)
        self.test_kepert_Vy = cPickle.load(pkl_file)
        self.test_mcconochie_Ux = cPickle.load(pkl_file)
        self.test_mcconochie_Vy = cPickle.load(pkl_file)
        self.test_hubbert_Ux = cPickle.load(pkl_file)
        self.test_hubbert_Vy = cPickle.load(pkl_file)
        pkl_file.close()

    def test_Kepert(self):
        profile = WindProfileModel(-15, 0.0, 1000., 1000., self.rMax, WindSpeedModel)
        import pdb; pdb.set_trace()
        profile.f = self.f
        windField = KepertWindField(profile)
        windField.V = self.V
        windField.Z = self.Z

        Ux, Vy = windField.field(self.R, self.lam, self.vFm, self.thetaFm, self.thetaMax)
        self.numpyAssertAlmostEqual(Ux, self.test_kepert_Ux)
        self.numpyAssertAlmostEqual(Vy, self.test_kepert_Vy)
        wndsp = np.sqrt(Ux ** 2 +  Vy ** 2)
        plt.figure()
        ax=plt.axes(projection=ccrs.PlateCarree())
        plotMap(wndsp,windField.V[0,:], windField.V[1,:], ax)
        plt.show()



if __name__ == "__main__":

    testSuite = unittest.makeSuite(TestWindField, 'test')
    unittest.TextTestRunner(verbosity=2).run(testSuite)
