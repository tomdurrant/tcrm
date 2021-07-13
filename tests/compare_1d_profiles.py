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
 Author: Tom Durrant, t.durrant@metocean.co.nz
 CreationDate: 2016-02-09
 Description: Unit testing module for pressureProfile

 Version: $Rev$

 $Id$
"""
import os, sys
import cPickle
import unittest
import NumpyTestCase
import matplotlib.pyplot as plt
import numpy as np
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
from wind.windmodels import *

class TestPressureProfile(NumpyTestCase.NumpyTestCase):

    pkl_file = open(os.path.join(unittest_dir, 'test_data', 'pressureProfileTestData.pck'), 'r')
    R = cPickle.load(pkl_file)
    R = np.arange(0,500)
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
    pHolland = prP.holland()
    pWilloughby = prP.willoughby()
    pdoubleHolland = prP.doubleHolland()
    pPowell = prP.powell()

    def test_pressure(self):
        """Testing pressure profile """
        plt.figure()
        plt.plot(self.R,self.pHolland,label='Holland')
        plt.plot(self.R,self.pWilloughby,label='Willoughby')
        plt.plot(self.R,self.pdoubleHolland,label='DoubleHolland')
        plt.plot(self.R,self.pPowell,label='Powell',)
        plt.legend(loc='lower right')
        plt.xlabel('Radial Distance (km)')
        plt.ylabel('Pressure (Pa)')

    def test_velocity(self):
        """Testing wind profile """
        plt.figure()
        profile = RankineWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.velocity(self.R)
        plt.plot(self.R,V,label='Rankine')

        profile = JelesnianskiWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.velocity(self.R)
        plt.plot(self.R,V,label='Jelesnianski')

        profile = HollandWindProfile(self.cLat, self.cLon, self.pEnv,
                                     self.pCentre, self.rMax, self.beta)
        V = profile.velocity(self.R)
        plt.plot(self.R,V,label='Holland')

        profile = WilloughbyWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.velocity(self.R)
        plt.plot(self.R,V,label='Willoughby')

        profile = PowellWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.velocity(self.R)
        plt.plot(self.R,V,label='Powell')

        profile = DoubleHollandWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax,
            self.beta1, self.beta2, self.rMax2)
        V = profile.velocity(self.R)
        plt.plot(self.R,V,label='DoubleHolland')

        profile =NewHollandWindProfile(
            self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax)
        V = profile.velocity(self.R)
        plt.plot(self.R,V,label='NewHolland')

        plt.legend(loc='lower right')
        plt.xlabel('Radial Distance (km)')
        plt.ylabel('Azimuthal Velocity (m/s)')

    def test_newHolland(self):
        """Testing Holland 2010 profile """
        plt.figure()
        for rGale in (150,200,250):
            profile =NewHollandWindProfile(
                self.cLat, self.cLon, self.pEnv, self.pCentre, self.rMax,rGale=rGale)
            V = profile.velocity(self.R)
            plt.plot(self.R,V,label=rGale)
        plt.legend()
        plt.legend(loc='upper right')
        plt.xlabel('Radial Distance (km)')
        plt.ylabel('Azimuthal Velocity (m/s)')


if __name__ == "__main__":
    flStartLog('', 'CRITICAL', False)
    testSuite = unittest.makeSuite(TestPressureProfile,'test')
    unittest.TextTestRunner(verbosity=2).run(testSuite)
    plt.show()
