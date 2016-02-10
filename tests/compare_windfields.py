from Utilities.plotting import plotMap
import xray as xr
import matplotlib.pyplot as plt
import NumpyTestCase
import unittest
import sys
import os
import cPickle
from wind.windmodels import *
import seaborn
seaborn.reset_orig()

try:
    import pathLocate
except:
    from tests import pathLocate

# Add parent folder to python path
unittest_dir = pathLocate.getUnitTestDirectory()
sys.path.append(pathLocate.getRootDirectory())

def plot(V,subplot=111,title=None,vmin=None,vmax=None):
    fig=plt.gcf()
    ax=fig.add_subplot(subplot)
    map=plt.imshow(V,cmap=plt.get_cmap('viridis'),vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.grid(True)
    plt.title(title)

def plotvec(Ux,Vy,subplot=111,title=None,vmin=None,vmax=None):
    fig=plt.gcf()
    ax=fig.add_subplot(subplot)
    map=plt.imshow(np.sqrt(Ux**2+Vy**2),cmap=plt.get_cmap('viridis'),vmin=vmin,vmax=vmax)
    plt.colorbar()
    plt.grid(True)
    plt.title(title)


class TestWindVelocity(NumpyTestCase.NumpyTestCase):
    vmax=0
    vmin=-50

    def setUp(self):
        pkl_file = open(os.path.join(
            unittest_dir, 'test_data', 'windProfileTestData.pck'), 'r')
        self.R = cPickle.load(pkl_file)
        self.pEnv = cPickle.load(pkl_file)
        self.pCentre = cPickle.load(pkl_file)
        self.rMax = cPickle.load(pkl_file)
        self.cLat = cPickle.load(pkl_file)
        self.cLon = cPickle.load(pkl_file)
        self.beta = cPickle.load(pkl_file)
        self.rMax2 = cPickle.load(pkl_file)
        self.beta1 = cPickle.load(pkl_file)
        self.beta2 = cPickle.load(pkl_file)
        self.test_wP_rankine = cPickle.load(pkl_file)
        self.test_wP_jelesnianski = cPickle.load(pkl_file)
        self.test_wP_holland = cPickle.load(pkl_file)
        self.test_wP_willoughby = cPickle.load(pkl_file)
        self.test_wP_powell = cPickle.load(pkl_file)
        self.test_wP_doubleHolland = cPickle.load(pkl_file)
        pkl_file.close()

    def testRankine(self):
        profile = RankineWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.velocity(self.R)
        plot(V,331,'Rankine',vmin=self.vmin,vmax=self.vmax)

    def testJelesnianski(self):
        profile = JelesnianskiWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.velocity(self.R)
        plot(V,332,'Jelesnianski',vmin=self.vmin,vmax=self.vmax)

    def testHolland(self):
        profile = HollandWindProfile(self.cLat, self.cLon, self.pEnv,
                                     self.pCentre, self.rMax, self.beta)
        V = profile.velocity(self.R)
        plot(V,333,'Holland',vmin=self.vmin,vmax=self.vmax)

    def testWilloughby(self):
        profile = WilloughbyWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.velocity(self.R)
        plot(V,334,'Willoughby',vmin=self.vmin,vmax=self.vmax)

    def testPowell(self):
        profile = PowellWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.velocity(self.R)
        plot(V,335,'Powell',vmin=self.vmin,vmax=self.vmax)

    def testDoubleHolland(self):
        profile = DoubleHollandWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax,
            self.beta1, self.beta2, self.rMax2)
        V = profile.velocity(self.R)
        plot(V,336,'doubleHolland',vmin=self.vmin,vmax=self.vmax)

    def testNewHolland(self):
        profile =NewHollandWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.velocity(self.R)
        plot(V,337,'newHolland',vmin=self.vmin,vmax=self.vmax)

    def testNewHolland100(self):
        profile =NewHollandWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax,rGale=100)
        V = profile.velocity(self.R)
        plot(V,338,'newHolland',vmin=self.vmin,vmax=self.vmax)


class TestWindVorticity(NumpyTestCase.NumpyTestCase):
    vmax=50
    vmin=-50

    def setUp(self):
        pkl_file = open(os.path.join(unittest_dir, 'test_data', 'vorticityTestData.pck'), 'r')
        self.R = cPickle.load(pkl_file)
        self.pEnv = cPickle.load(pkl_file)
        self.pCentre = cPickle.load(pkl_file)
        self.rMax = cPickle.load(pkl_file)
        self.cLat = cPickle.load(pkl_file)
        self.cLon = cPickle.load(pkl_file)
        self.beta = cPickle.load(pkl_file)
        self.rMax2 = cPickle.load(pkl_file)
        self.beta1 = cPickle.load(pkl_file)
        self.beta2 = cPickle.load(pkl_file)
        self.vMax = cPickle.load(pkl_file)
        self.test_vorticity_rankine = cPickle.load(pkl_file)
        self.test_vorticity_jelesnianski = cPickle.load(pkl_file)
        self.test_vorticity_holland = cPickle.load(pkl_file)
        self.test_vorticity_willoughby = cPickle.load(pkl_file)
        self.test_vorticity_doubleHolland = cPickle.load(pkl_file)
        self.test_vorticity_powell = cPickle.load(pkl_file)
        pkl_file.close()

    def testRankine(self):
        profile = RankineWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        profile.vMax = self.vMax
        V = profile.vorticity(self.R)
        plot(V,331,'Rankine',vmin=self.vmin,vmax=self.vmax)

    def testJelesnianski(self):
        profile = JelesnianskiWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        profile.vMax = self.vMax
        V = profile.vorticity(self.R)
        plot(V,332,'Jelesnianski',vmin=self.vmin,vmax=self.vmax)

    def testHolland(self):
        profile = HollandWindProfile(self.cLat, self.cLon, self.pEnv,
                                     self.pCentre, self.rMax, self.beta)
        profile.vMax = self.vMax
        V = profile.vorticity(self.R)
        plot(V,333,'Holland',vmin=self.vmin,vmax=self.vmax)

    def testWilloughby(self):
        profile = WilloughbyWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        # Hack for testing as vMax needs to be set
        profile.vMax = self.vMax
        profile.beta = 1.0036 + 0.0173 * profile.vMax - 0.313 * np.log(self.rMax) + 0.0087 * np.abs(self.cLat)
        V = profile.vorticity(self.R)
        plot(V,334,'Willoughby',vmin=self.vmin,vmax=self.vmax)

    def testDoubleHolland(self):
        profile = DoubleHollandWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax,
            self.beta1, self.beta2, self.rMax2)
        profile.vMax = self.vMax
        V = profile.vorticity(self.R)
        plot(V,335,'DoubleHolland',vmin=self.vmin,vmax=self.vmax)

    def testPowell(self):
        profile = PowellWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        profile.vMax = self.vMax
        V = profile.vorticity(self.R)
        plot(V,336,'Powell',vmin=self.vmin,vmax=self.vmax)


class TestWindField(NumpyTestCase.NumpyTestCase):
    vmin=0
    vmax=50

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
        profile.f = self.f
        windField = KepertWindField(profile)
        windField.V = self.V
        windField.Z = self.Z

        Ux, Vy = windField.field(self.R, self.lam, self.vFm, self.thetaFm, self.thetaMax)
        plotvec(Ux,Vy,231,title='Kepert',vmin=self.vmin,vmax=self.vmax)

    def test_McConochie(self):
        profile = WindProfileModel(0.0, 0.0, 1000., 1000., self.rMax, WindSpeedModel)
        profile.f = self.f
        windField = McConochieWindField(profile)
        windField.V = self.V
        windField.Z = self.Z
        Ux, Vy = windField.field(self.R, self.lam, self.vFm, self.thetaFm, self.thetaMax)
        plotvec(Ux,Vy,232,title='McConochie',vmin=self.vmin,vmax=self.vmax)

    def test_Hubbert(self):
        profile = WindProfileModel(0.0, 0.0, 1000., 1000., self.rMax, WindSpeedModel)
        profile.f = self.f
        windField = HubbertWindField(profile)
        windField.V = self.V
        windField.Z = self.Z
        Ux, Vy = windField.field(self.R, self.lam, self.vFm, self.thetaFm, self.thetaMax)
        plotvec(Ux,Vy,233,title='Hubbert',vmin=self.vmin,vmax=self.vmax)

if __name__ == "__main__":
    # fig=plt.figure(figsize=(12,9))
    # testSuite = unittest.makeSuite(TestWindVelocity, 'test')
    # unittest.TextTestRunner(verbosity=2).run(testSuite)

    # fig=plt.figure(figsize=(12,9))
    # testSuite = unittest.makeSuite(TestWindVorticity, 'test')
    # unittest.TextTestRunner(verbosity=2).run(testSuite)
    # plt.show()

    fig=plt.figure(figsize=(12,6))
    testSuite = unittest.makeSuite(TestWindField, 'test')
    unittest.TextTestRunner(verbosity=2).run(testSuite)
    plt.show()

