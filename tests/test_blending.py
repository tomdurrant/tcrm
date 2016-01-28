import os
import sys
import unittest
import cPickle
import NumpyTestCase
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from wind.windmodels import *
from plotting import plotMap

try:
    import pathLocate
except:
    from tests import pathLocate

# Add parent folder to python path
unittest_dir = pathLocate.getUnitTestDirectory()
sys.path.append(pathLocate.getRootDirectory())

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
        profile = WindProfileModel(-15, 0.0, 980., 1000., self.rMax, WindSpeedModel)
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
        import pdb; pdb.set_trace()
        plotMap(wndsp,windField.V[0,:], windField.V[1,:], ax)
        plt.show()



if __name__ == "__main__":

    testSuite = unittest.makeSuite(TestWindField, 'test')
    unittest.TextTestRunner(verbosity=2).run(testSuite)
