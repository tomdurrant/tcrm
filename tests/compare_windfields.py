# from Utilities.plotting import plotMap
import xarray as xr
import matplotlib.pyplot as plt
import NumpyTestCase
import unittest
import sys
import os
import pickle
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
        self.cLon = 158.17
        self.cLat = -20.13
        self.pCentre = 95330.
        self.pEnv = 101445.0
        self.rMax = 50000.
        self.rMax2 = 90000.
        self.beta = 1.7
        self.beta1 = 1.7
        self.beta2 = 1.3
        self.vFm = 10.
        self.thetaFm = 70. * np.pi / 180.
        self.thetaMax = 70. * np.pi / 180.

        pkl_file = open(os.path.join(
            unittest_dir, 'test_data', 'windProfileTestData.pkl'), 'rb')
        self.R = pickle.load(pkl_file)
        self.lam = pickle.load(pkl_file)
        self.test_wP_rankine = pickle.load(pkl_file)
        self.test_wP_jelesnianski = pickle.load(pkl_file)
        self.test_wP_holland = pickle.load(pkl_file)
        self.test_wP_willoughby = pickle.load(pkl_file)
        self.test_wP_powell = pickle.load(pkl_file)
        self.test_wP_doubleHolland = pickle.load(pkl_file)
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
        self.cLon = 158.17
        self.cLat = -20.13
        self.pCentre = 95330.
        self.pEnv = 101445.0
        self.rMax = 50000.
        self.rMax2 = 90000.
        self.beta = 1.7
        self.beta1 = 1.7
        self.beta2 = 1.3
        self.vFm = 10.
        self.thetaFm = 70. * np.pi / 180.
        self.thetaMax = 70. * np.pi / 180.
        self.profile = HollandWindProfile(self.cLat, self.cLon, self.pEnv,
                                          self.pCentre, self.rMax, self.beta)

        pkl_file = open(os.path.join(unittest_dir, 'test_data',
                        'vorticityTestData.pkl'), 'rb')
        self.R = pickle.load(pkl_file)
        self.lam = pickle.load(pkl_file)
        self.test_vorticity_rankine = pickle.load(pkl_file)
        self.test_vorticity_jelesnianski = pickle.load(pkl_file)
        self.test_vorticity_holland = pickle.load(pkl_file)
        self.test_vorticity_willoughby = pickle.load(pkl_file)
        self.test_vorticity_doubleHolland = pickle.load(pkl_file)
        self.test_vorticity_powell = pickle.load(pkl_file)
        pkl_file.close()

    def testRankine(self):
        profile = RankineWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.vorticity(self.R)
        plot(V,331,'Rankine',vmin=self.vmin,vmax=self.vmax)

    def testJelesnianski(self):
        profile = JelesnianskiWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.vorticity(self.R)
        plot(V,332,'Jelesnianski',vmin=self.vmin,vmax=self.vmax)

    def testHolland(self):
        profile = HollandWindProfile(self.cLat, self.cLon, self.pEnv,
                                     self.pCentre, self.rMax, self.beta)
        V = profile.vorticity(self.R)
        plot(V,333,'Holland',vmin=self.vmin,vmax=self.vmax)

    def testWilloughby(self):
        profile = WilloughbyWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        profile.beta = 1.0036 + 0.0173 * profile.vMax - \
            0.0313 * np.log(self.rMax) + 0.0087 * np.abs(self.cLat)
        V = profile.vorticity(self.R)
        plot(V,334,'Willoughby',vmin=self.vmin,vmax=self.vmax)

    def testDoubleHolland(self):
        profile = DoubleHollandWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax,
            self.beta1, self.beta2, self.rMax2)
        V = profile.vorticity(self.R)
        plot(V,335,'DoubleHolland',vmin=self.vmin,vmax=self.vmax)

    def testPowell(self):
        profile = PowellWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.vorticity(self.R)
        plot(V,336,'Powell',vmin=self.vmin,vmax=self.vmax)


class TestWindField(NumpyTestCase.NumpyTestCase):
    vmin=0
    vmax=50

    def setUp(self):
        self.cLon = 158.17
        self.cLat = -20.13
        self.pCentre = 95330.
        self.pEnv = 101445.0
        self.rMax = 50000.
        self.rMax2 = 90000.
        self.beta = 1.7
        self.beta1 = 1.7
        self.beta2 = 1.3
        self.vFm = 10.
        self.thetaFm = 70. * np.pi / 180.
        self.thetaMax = 70. * np.pi / 180.
        self.profile = HollandWindProfile(self.cLat, self.cLon, self.pEnv,
                                          self.pCentre, self.rMax, self.beta)

        pkl_file = open(os.path.join(unittest_dir, 'test_data',
                        'windFieldTestData.pkl'), 'rb')
        self.R = pickle.load(pkl_file)
        self.lam = pickle.load(pkl_file)
        self.test_kepert_Ux = pickle.load(pkl_file)
        self.test_kepert_Vy = pickle.load(pkl_file)
        self.test_mcconochie_Ux = pickle.load(pkl_file)
        self.test_mcconochie_Vy = pickle.load(pkl_file)
        self.test_hubbert_Ux = pickle.load(pkl_file)
        self.test_hubbert_Vy = pickle.load(pkl_file)
        pkl_file.close()

    def test_Kepert(self):
        windField = KepertWindField(self.profile)
        Ux, Vy = windField.field(self.R, self.lam, self.vFm, self.thetaFm, self.thetaMax)
        plotvec(Ux,Vy,231,title='Kepert',vmin=self.vmin,vmax=self.vmax)

    def test_McConochie(self):
        windField = McConochieWindField(self.profile)
        Ux, Vy = windField.field(self.R, self.lam, self.vFm, self.thetaFm,
                                 self.thetaMax)
        plotvec(Ux,Vy,232,title='McConochie',vmin=self.vmin,vmax=self.vmax)

    def test_Hubbert(self):
        windField = HubbertWindField(self.profile)
        Ux, Vy = windField.field(self.R, self.lam, self.vFm, self.thetaFm,
                                 self.thetaMax)
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

