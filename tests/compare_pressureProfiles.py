"""
    Tropical Cyclone Risk Model (TCRM) - Version 1.0
    Copyright (C) 2011 Commonwealth of Australia (Geoscience Australia)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


 Title: TestPressureProfile.py
 Author: Nicholas Summons, nicholas.summons@ga.gov.au
 CreationDate: 2011-06-09
 Description: Unit testing module for pressureProfile

 Version: $Rev$

 $Id$
"""
import os, sys
import cPickle
import unittest
import NumpyTestCase
import matplotlib.pyplot as plt
import seaborn
seaborn.reset_orig()

try:
    import pathLocate
except:
    from unittests import pathLocate

# Add parent folder to python path
unittest_dir = pathLocate.getUnitTestDirectory()
sys.path.append(pathLocate.getRootDirectory())
from PressureInterface import pressureProfile
from Utilities.files import flStartLog

class TestPressureProfile(NumpyTestCase.NumpyTestCase):

    pkl_file = open(os.path.join(unittest_dir, 'test_data', 'pressureProfileTestData.pck'), 'r')
    R = cPickle.load(pkl_file)
    pEnv = cPickle.load(pkl_file)
    pCentre = cPickle.load(pkl_file)
    rMax = cPickle.load(pkl_file)
    cLat = cPickle.load(pkl_file)
    cLon = cPickle.load(pkl_file)
    beta = cPickle.load(pkl_file)
    rMax2 = cPickle.load(pkl_file)
    beta1 = cPickle.load(pkl_file)
    beta2 = cPickle.load(pkl_file)
    test_pHolland = cPickle.load(pkl_file)
    test_pWilloughby = cPickle.load(pkl_file)
    test_pdoubleHolland = cPickle.load(pkl_file)
    test_pPowell = cPickle.load(pkl_file)
    pkl_file.close()

    prP = pressureProfile.PrsProfile(R, pEnv, pCentre, rMax, cLat, cLon, beta, rMax2, beta1, beta2)
    plt.figure(figsize=(12,9))

    def plot(self,V,subplot=111,title=None,vmin=95400,vmax=100200):
        fig=plt.gcf()
        fig.add_subplot(subplot)
        map=plt.imshow(V,cmap=plt.get_cmap('viridis'),vmin=vmin,vmax=vmax)
        plt.colorbar()
        plt.grid(True)
        plt.title(title)

    def test_Holland(self):
        """Testing Holland profile """
        pHolland = self.prP.holland()
        self.plot(pHolland,331,'Holland')

    def test_Willoughby(self):
        """Testing Willoughby profile """
        pWilloughby = self.prP.willoughby()
        self.plot(pWilloughby,332,'Willoughby')

    def test_doubleHolland(self):
        """Testing Double Holland profile """
        pdoubleHolland = self.prP.doubleHolland()
        self.plot(pdoubleHolland,333,'doubleHolland')

    def test_Powell(self):
        """Testing Powell profile """
        pPowell = self.prP.powell()
        self.plot(pPowell,334,'Powell')

    def test_Powell(self):
        """Testing Powell profile """
        pPowell = self.prP.powell()
        self.plot(pPowell,334,'Powell')

if __name__ == "__main__":
    flStartLog('', 'CRITICAL', False)
    testSuite = unittest.makeSuite(TestPressureProfile,'test')
    unittest.TextTestRunner(verbosity=2).run(testSuite)
    plt.show()
