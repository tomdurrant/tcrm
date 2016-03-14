"""
:mod:`wind` -- Wind field calculation
=====================================

This module contains the core object for the wind field
calculations. It provides the radial profile models to define the
primary vortex of the simulated TC, and bounday layer models that
define the asymmetry induced by surface friction and forward motion of
the TC over the earth's surface. The final output from the module is a
netCDF file containing the maximum surface gust wind speed (a 10-minute 
mean wind speed, at 10 metres above ground level), along with the components
(eastward and westward) that generated the wind gust and the minimum
mean sea level pressure over the lifetime of the event. If multiple
TCs are contained in a track file, then the output file contains the
values from all events (for example, an annual maximum wind speed).

Wind field calculations can be run in parallel using MPI if the
:term:`pypar` library is found and TCRM is run using the
:term:`mpirun` command. For example, to run with 10 processors::

    $ mpirun -n 10 python tcrm.py cairns.ini

:class:`wind` can be correctly initialised and started by
calling the :meth:`run` with the location of a *configFile*::

    >>> import wind
    >>> wind.run('cairns.ini')

"""

import xray as xr
from netCDF4 import Dataset, date2num
import numpy as np
import numpy.ma as ma
import logging as log
import itertools
import math
import os
import sys
import windmodels
from datetime import datetime
import time as timemod
from os.path import join as pjoin, split as psplit, splitext as psplitext
from collections import defaultdict

from PlotInterface.maps import saveWindfieldMap

from Utilities.files import flModDate, flProgramVersion
from Utilities.config import ConfigParser
from Utilities.metutils import convert
from Utilities.maputils import bearing2theta, makeGrid
from Utilities.parallel import attemptParallel

import Utilities.nctools as nctools

from Utilities.track import ncReadTrackData, Track

class WindfieldAroundTrack(object):
    """
    The windfield around the tropical cyclone track.


    :type  track: :class:`Track`
    :param track: the tropical cyclone track.

    :type  profileType: str
    :param profileType: the wind profile type.

    :type  windFieldType: str
    :param windFieldType: the wind field type.

    :type  beta: float
    :param beta: wind field parameter.

    :type  beta1: float
    :param beta1: wind field parameter.

    :type  beta2: float
    :param beta2: wind field parameter.

    :type  thetaMax: float
    :param thetaMax:

    :type  margin: float
    :param margin:

    :type  resolution: float
    :param resolution: Grid resolution (in degrees)

    :type  gustFactor: float
    :param gustFactor: Conversion from 1-min mean to 0.2-sec gust wind speed.

    :type  gridLimit: :class:`dict`
    :param gridLimit: the domain where the tracks will be generated.
                      The :class:`dict` should contain the keys
                      :attr:`xMin`, :attr:`xMax`, :attr:`yMin` and
                      :attr:`yMax`. The *y* variable bounds the
                      latitude and the *x* variable bounds the
                      longitude.

    """

    def __init__(self, track, config, profileType='powell', windFieldType='kepert',
                 beta=1.3, beta1=1.5, beta2=1.4, thetaMax=70.0, margin=2.0,
                 resolution=0.05, gustFactor=1.188, gridLimit=None,
                 domain='bounded'):
        self.track = track
        self.profileType = profileType
        self.windFieldType = windFieldType
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.thetaMax = math.radians(thetaMax)
        self.margin = margin
        self.resolution = resolution
        self.gustFactor = gustFactor
        self.gridLimit = gridLimit
        self.domain = domain
        self.config = config

    def polarGridAroundEye(self, i):
        """
        Generate a polar coordinate grid around the eye of the
        tropical cyclone at time i.

        :type  i: int
        :param i: the time.
        """
        if self.domain == 'full':
            R, theta = makeGrid(self.track.Longitude[i],
                                self.track.Latitude[i],
                                self.margin, self.resolution,
                                minLon=self.gridLimit['xMin'],
                                maxLon=self.gridLimit['xMax'],
                                minLat=self.gridLimit['yMin'],
                                maxLat=self.gridLimit['yMax'])
        else:
            R, theta = makeGrid(self.track.Longitude[i],
                                self.track.Latitude[i],
                                self.margin, self.resolution)
        return R, theta

    def pressureProfile(self, i, R):
        """
        Calculate the pressure profile at time `i` at the radiuses `R`
        around the tropical cyclone.


        :type  i: int
        :param i: the time.

        :type  R: :class:`numpy.ndarray`
        :param R: the radiuses around the tropical cyclone.
        """
        from PressureInterface.pressureProfile import PrsProfile as PressureProfile

        p = PressureProfile(R, convert(self.track.EnvPressure[i], 'hPa', 'Pa'),
                            convert(self.track.CentralPressure[i], 'hPa', 'Pa'),
                            self.track.rMax[i],
                            self.track.Latitude[i],
                            self.track.Longitude[i],
                            self.beta, beta1=self.beta1,
                            beta2=self.beta2)
        try:
            pressure = getattr(p, self.profileType)
        except AttributeError:
            msg = '%s not implemented in pressureProfile' % self.profileType
            log.exception(msg)
        return pressure()

    def localWindField(self, i):
        """
        Calculate the local wind field at time `i` around the
        tropical cyclone.

        :type  i: int
        :param i: the time.
        """
        lat = self.track.Latitude[i]
        lon = self.track.Longitude[i]
        eP = convert(self.track.EnvPressure[i], 'hPa', 'Pa')
        cP = convert(self.track.CentralPressure[i], 'hPa', 'Pa')
        rMax = self.track.rMax[i]
        self.rGale = self.track.rGale[i]
        vFm = convert(self.track.Speed[i], 'kmh', 'mps')
        thetaFm = bearing2theta(self.track.Bearing[i] * np.pi/180.),
        thetaMax = self.thetaMax

        #FIXME: temporary way to do this
        cls = windmodels.profile(self.profileType)
        params = windmodels.profileParams(self.profileType)
        values = [getattr(self, p) for p in params if hasattr(self, p)]
        profile = cls(lat, lon, eP, cP, rMax, *values)

        R, theta = self.polarGridAroundEye(i)

        P = self.pressureProfile(i, R)

        #FIXME: temporary way to do this
        cls = windmodels.field(self.windFieldType)
        params = windmodels.fieldParams(self.windFieldType)
        values = [getattr(self, p) for p in params if hasattr(self, p)]
        windfield = cls(profile, *values)

        Ux, Vy = windfield.field(R, theta, vFm, thetaFm,  thetaMax)

        blendWinds = self.config.getboolean('WindfieldInterface', 'blendWinds')
        blendWindsMethod = self.config.get('WindfieldInterface', 'blendWindsMethod')
        radiusFactor = self.config.getfloat('WindfieldInterface', 'radiusFactor')
        if blendWinds:
            from blend import blendWeights
            if blendWindsMethod == 'rGale':
                sig = self.rGale * radiusFactor
            elif blendWindsMethod == 'rMax':
                sig = rMax * radiusFactor
            else:
                raise Exception("Blendmethod %s not recognised" % blendWindsMethod)
            bweights = blendWeights(R,sig)
        else:
            bweights = None

        return (Ux, Vy, P, bweights)

    def regionalExtremes(self, gridLimit, timeStepCallback=None,):
        """
        Calculate the maximum potential wind gust and minimum
        pressure over the region throughout the life of the
        tropical cyclone.


        :type  gridLimit: :class:`dict`
        :param gridLimit: the domain where the tracks will be considered.
                          The :class:`dict` should contain the keys
                          :attr:`xMin`, :attr:`xMax`, :attr:`yMin` and
                          :attr:`yMax`. The *y* variable bounds the
                          latitude and the *x* variable bounds the longitude.

        :type  timeStepCallback: function
        :param timeStepCallback: the function to be called on each time step.
        """
        writeWinds = self.config.getboolean('WindfieldInterface', 'writeWinds')
        blendWinds = self.config.getboolean('WindfieldInterface', 'blendWinds')
        if len(self.track.data) > 0:
            envPressure = convert(self.track.EnvPressure[0], 'hPa', 'Pa')
        else:
            envPressure = np.NaN

        # Get the limits of the region
        xMin = gridLimit['xMin']
        xMax = gridLimit['xMax']
        yMin = gridLimit['yMin']
        yMax = gridLimit['yMax']

        # Setup a 'millidegree' integer grid for the region
        gridMargin = int(100. * self.margin)
        gridStep = int(100. * self.resolution)

        minLat = int(100. * yMin) - gridMargin
        maxLat = int(100. * yMax) + gridMargin
        minLon = int(100. * xMin) - gridMargin
        maxLon = int(100. * xMax) + gridMargin

        latGrid = np.arange(minLat, maxLat + gridStep, gridStep, dtype=int)
        lonGrid = np.arange(minLon, maxLon + gridStep, gridStep, dtype=int)

        [cGridX, cGridY] = np.meshgrid(lonGrid, latGrid)

        # Initialise the region
        UU = np.zeros_like(cGridX, dtype='f')
        VV = np.zeros_like(cGridY, dtype='f')
        bearing = np.zeros_like(cGridX, dtype='f')
        gust = np.zeros_like(cGridX, dtype='f')
        pressure = np.ones_like(cGridX, dtype='f') * envPressure

        lonCDegree = np.array(100. * self.track.Longitude, dtype=int)
        latCDegree = np.array(100. * self.track.Latitude, dtype=int)

        # We only consider the times when the TC track falls in the region
        timesInRegion = np.where((xMin <= self.track.Longitude) &
                                (self.track.Longitude <= xMax) &
                                (yMin <= self.track.Latitude) &
                                (self.track.Latitude <= yMax))[0]


        if writeWinds:
            dtout = self.config.getfloat('WindfieldInterface', 'dtout')
            outputPath = self.config.get('Output', 'Path')
            windfieldPath = pjoin(outputPath, 'windfield')

            vortex = pjoin(windfieldPath,
                             'vortex.{0:03d}-{1:05d}.nc'.\
                             format(*self.track.trackId))

            vortexdset = create_nc(vortex,latGrid / 100.,lonGrid / 100.,
                                   self.track.trackfile, self.config)

            if blendWinds:
                windblend = pjoin(windfieldPath,
                                 'blended.{0:03d}-{1:05d}.nc'.\
                                 format(*self.track.trackId))

                vars=['mslp','uwnd','vwnd','bw']
                descs=['Pressure','uwind','vwnd','blend_weights']
                units=['hPa','ms','ms','test']
                blenddset = create_nc(windblend, latGrid / 100., lonGrid /
                                      100., self.track.trackfile, self.config,
                                      vars=vars, descs=descs, units=units)

                background = pjoin(windfieldPath,
                                 'background.{0:03d}-{1:05d}.nc'.\
                                 format(*self.track.trackId))

                from blend import getData
                self.data = getData(self.track.Datetime[timesInRegion[0]],
                                    self.track.Datetime[timesInRegion[-1]],
                                    var=['ugrd10m','vgrd10m','mslp'],
                                    dset=['cfsr'],
                                    outnc = background,
                                    bnd=[lonGrid.min()/100.,lonGrid.max()/100.,latGrid.min()/100.,latGrid.max()/100.],
                                    res=self.resolution, dt=dtout,
                                    udshost='http://uds1.rag.metocean.co.nz:9191/uds')

        for i in timesInRegion:

            uwnd = np.zeros((latGrid.size,lonGrid.size), dtype='f')
            vwnd = np.zeros((latGrid.size,lonGrid.size), dtype='f')
            mslp = np.ones((latGrid.size,lonGrid.size), dtype='f') * envPressure
            bweights = np.zeros((latGrid.size,lonGrid.size), dtype='f')

            # Map the local grid to the regional grid
            jmin, jmax = 0, int((maxLat - minLat + 2. * gridMargin) \
                                / gridStep) + 1
            imin, imax = 0, int((maxLon - minLon + 2. * gridMargin) \
                                / gridStep) + 1

            if self.domain == 'bounded':

                jmin = int((latCDegree[i] - minLat - gridMargin) / gridStep)
                jmax = int((latCDegree[i] - minLat + gridMargin) / gridStep) + 1
                imin = int((lonCDegree[i] - minLon - gridMargin) / gridStep)
                imax = int((lonCDegree[i] - minLon + gridMargin) / gridStep) + 1

            # Calculate the local wind speeds and pressure at time i

            Ux, Vy, P, bw = self.localWindField(i)

            # Calculate the local wind gust and bearing
            Uxg = Ux * self.gustFactor
            Vyg = Ux * self.gustFactor

            localGust = np.sqrt(Uxg ** 2 + Vyg ** 2)
            localBearing = ((np.arctan2(-Uxg, -Vyg)) * 180. / np.pi)

            # Handover this time step to a callback if required
            if timeStepCallback is not None:
                timeStepCallback(self.track.Datetime[i],
                                 localGust, Uxg, Vyg, P,
                                 lonGrid[imin:imax] / 100.,
                                 latGrid[jmin:jmax] / 100.)

            # Retain when there is a new maximum gust
            mask = localGust > gust[jmin:jmax, imin:imax]

            gust[jmin:jmax, imin:imax] = np.where(
                mask, localGust, gust[jmin:jmax, imin:imax])
            bearing[jmin:jmax, imin:imax] = np.where(
                mask, localBearing, bearing[jmin:jmax, imin:imax])
            UU[jmin:jmax, imin:imax] = np.where(
                mask, Uxg, UU[jmin:jmax, imin:imax])
            VV[jmin:jmax, imin:imax] = np.where(
                mask, Vyg, VV[jmin:jmax, imin:imax])

            # Retain the lowest pressure
            pressure[jmin:jmax, imin:imax] = np.where(
                P < pressure[jmin:jmax, imin:imax],
                P, pressure[jmin:jmax, imin:imax])

            # Write u and v where valid
            if writeWinds:
                mslp[jmin:jmax, imin:imax] = P
                uwnd[jmin:jmax, imin:imax] = Ux
                vwnd[jmin:jmax, imin:imax] = Vy
                rectime = self.track.Datetime[i]
                write_record(vortex,rectime,{'mslp':mslp, 'uwnd':uwnd,
                                             'vwnd':vwnd})

                if blendWinds:
                    bweights[jmin:jmax, imin:imax] = bw
                    bground = self.data.dset.sel(time = rectime, method='nearest')
                    if np.allclose(bground.lat.values, latGrid / 100.):
                        bground.coords['lat'] = latGrid / 100.
                    if np.allclose(bground.lon.values, lonGrid / 100.):
                        bground.coords['lon'] = lonGrid / 100.
                    # bground.to_netcdf('bgound.nc')
                    #bground = bground.rename({'mslp': 'mslp_bg', 'ugrd10m': 'uwnd_bg', 'vgrd10m': 'vwnd_bg'})
                    #dset = dset.rename({'mslp': 'mslp_tc', 'uwnd': 'uwnd_tc', 'vwnd': 'vwnd_tc'})
                    #dset.update(bground)
                    mslpb = (mslp * bweights) + (bground.mslp * (1 - bweights))
                    uwndb = (uwnd * bweights) + (bground.ugrd10m * (1 - bweights))
                    vwndb = (vwnd * bweights) + (bground.vgrd10m * (1 - bweights))
                    write_record(windblend, rectime, {'mslp':mslpb.values,
                                                      'uwnd':uwndb.values,
                                                      'vwnd':vwndb.values,
                                                      'bw':bweights})

        return gust, bearing, UU, VV, pressure, lonGrid / 100., latGrid / 100.




class WindfieldGenerator(object):
    """
    The wind field generator.


    :type  margin: float
    :param margin:

    :type  resolution: float
    :param resolution:

    :type  profileType: str
    :param profileType: the wind profile type.

    :type  windFieldType: str
    :param windFieldType: the wind field type.

    :type  beta: float
    :param beta: wind field parameter.

    :type  beta1: float
    :param beta1: wind field parameter.

    :type  beta2: float
    :param beta2: wind field parameter.

    :type  thetaMax: float
    :param thetaMax:

    :type  gridLimit: :class:`dict`
    :param gridLimit: the domain where the tracks will be generated.
                      The :class:`dict` should contain the keys :attr:`xMin`,
                      :attr:`xMax`, :attr:`yMin` and :attr:`yMax`. The *y*
                      variable bounds the latitude and the *x* variable bounds
                      the longitude.

    """

    def __init__(self, config, margin=2.0, resolution=0.05,
                 profileType='powell', windFieldType='kepert',
                 beta=1.5, beta1=1.5, beta2=1.4,
                 thetaMax=70.0, gridLimit=None, domain='bounded'):

        self.config = config
        self.margin = margin
        self.resolution = resolution
        self.profileType = profileType
        self.windFieldType = windFieldType
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.thetaMax = thetaMax
        self.gridLimit = gridLimit
        self.domain = domain

    def setGridLimit(self, track):
        """
        Set the outer bounds of the grid to encapsulate
        the extent of a single TC track.

        :param track: :class:`Track` object.

        """

        track_limits = {'xMin':9999, 'xMax':-9999, 'yMin':9999, 'yMax':-9999}
        track_limits['xMin'] = min(track_limits['xMin'], track.Longitude.min())
        track_limits['xMax'] = max(track_limits['xMax'], track.Longitude.max())
        track_limits['yMin'] = min(track_limits['yMin'], track.Latitude.min())
        track_limits['yMax'] = max(track_limits['yMax'], track.Latitude.max())
        self.gridLimit = {}
        self.gridLimit['xMin'] = np.floor(track_limits['xMin'])
        self.gridLimit['xMax'] = np.ceil(track_limits['xMax'])
        self.gridLimit['yMin'] = np.floor(track_limits['yMin'])
        self.gridLimit['yMax'] = np.ceil(track_limits['yMax'])


    def calculateExtremesFromTrack(self, track, callback=None):
        """
        Calculate the wind extremes given a single tropical cyclone track.


        :type  track: :class:`Track`
        :param track: the tropical cyclone track.

        :type  callback: function
        :param callback: optional function to be called at each timestep to
                         extract point values for specified locations.

        """
        if self.gridLimit is None:
            self.setGridLimit(track)

        wt = WindfieldAroundTrack(track,
                                  self.config,
                                  profileType=self.profileType,
                                  windFieldType=self.windFieldType,
                                  beta=self.beta,
                                  beta1=self.beta1,
                                  beta2=self.beta2,
                                  thetaMax=self.thetaMax,
                                  margin=self.margin,
                                  resolution=self.resolution,
                                  gridLimit=self.gridLimit,
                                  domain=self.domain)

        return track, wt.regionalExtremes(self.gridLimit, callback)


    def calculateExtremesFromTrackfile(self, trackfile, callback=None):
        """
        Calculate the wind extremes from a `trackfile` that might contain a
        number of tropical cyclone tracks. The wind extremes are calculated
        over the tracks, i.e., the maximum gusts and minimum pressures over all
        tracks are retained.


        :type  trackfile: str
        :param trackfile: the file name of the trackfile.

        :type  callback: function
        :param callback: optional function to be called at each timestep to
                         extract point values for specified locations.

        """
        trackiter = loadTracks(trackfile)
        f = self.calculateExtremesFromTrack

        results = (f(track, callback)[1] for track in trackiter)

        gust, bearing, Vx, Vy, P, lon, lat, dset = results.next()

        for result in results:
            gust1, bearing1, Vx1, Vy1, P1, lon1, lat1 = result
            gust = np.where(gust1 > gust, gust1, gust)
            Vx = np.where(gust1 > gust, Vx1, Vx)
            Vy = np.where(gust1 > gust, Vy1, Vy)
            P = np.where(P1 < P, P1, P)

        return (gust, bearing, Vx, Vy, P, lon, lat)

    def dumpGustsFromTracks(self, trackiter, windfieldPath,
                            timeStepCallback=None):
        """
        Dump the maximum wind speeds (gusts) observed over a region to
        netcdf files. One file is created for every track file.

        :type  trackiter: list of :class:`Track` objects
        :param trackiter: a list of :class:`Track` objects.

        :type  windfieldPath: str
        :param windfieldPath: the path where to store the gust output files.

        :type  filenameFormat: str
        :param filenameFormat: the format string for the output file names. The
                               default is set to 'gust-%02i-%04i.nc'.

        :type  timeStepCallBack: function
        :param timeStepCallback: optional function to be called at each
                                 timestep to extract point values for
                                 specified locations.
        """
        if timeStepCallback:
            results = itertools.imap(self.calculateExtremesFromTrack,
                                     trackiter,
                                     itertools.repeat(timeStepCallback))
        else:
            results = itertools.imap(self.calculateExtremesFromTrack,
                                     trackiter)

        for track, result in results:
            log.debug("Saving data for track {0:03d}-{1:05d}"\
                      .format(*track.trackId))
            gust, bearing, Vx, Vy, P, lon, lat = result

            dumpfile = pjoin(windfieldPath,
                             'gust.{0:03d}-{1:05d}.nc'.\
                             format(*track.trackId))
            plotfile = pjoin(windfieldPath,
                             'gust.{0:03d}-{1:05d}.png'.\
                             format(*track.trackId))
            self.saveGustToFile(track.trackfile,
                                (lat, lon, gust, Vx, Vy, P),
                                dumpfile)
            windfile = pjoin(windfieldPath,
                             'winds.{0:03d}-{1:05d}.nc'.\
                             format(*track.trackId))
            #self.saveWindToFile(track.trackfile,dset,windfile)
            #self.plotGustToFile((lat, lon, gust, Vx, Vy, P), plotfile)

    def plotGustToFile(self, result, filename):
        """
        Plot the wind field on a map
        """
        lat, lon, speed, Vx, Vy, P = result
        mapkwargs = dict(llcrnrlon=self.gridLimit['xMin'],
                         llcrnrlat=self.gridLimit['yMin'],
                         urcrnrlon=self.gridLimit['xMax'],
                         urcrnrlat=self.gridLimit['yMax'],
                         resolution='i',
                         projection='merc')
        levels = np.arange(20., 100.1, 5.)
        cbarlabel = 'Wind speed (m/s)'
        [gx, gy] = np.meshgrid(lon, lat)
        title = 'TC wind field'
        saveWindfieldMap(speed, gx, gy, title, levels,
                         cbarlabel, mapkwargs, filename)

    def saveGustToFile(self, trackfile, result, filename):
        """
        Save gusts to a file.
        """
        lat, lon, speed, Vx, Vy, P = result

        trackfileDate = flModDate(trackfile)

        gatts = {
            'title': 'TCRM hazard simulation - synthetic event wind field',
            'tcrm_version': flProgramVersion(),
            'python_version': sys.version,
            'track_file': trackfile,
            'track_file_date': trackfileDate,
            'radial_profile': self.profileType,
            'boundary_layer': self.windFieldType,
            'beta': self.beta}

        # Add configuration settings to global attributes:
        for section in self.config.sections():
            for option in self.config.options(section):
                key = "{0}_{1}".format(section, option)
                value = self.config.get(section, option)
                gatts[key] = value

        dimensions = {
            0: {
                'name': 'lat',
                'values': lat,
                'dtype': 'f',
                'atts': {
                    'long_name': 'Latitude',
                    'standard_name': 'latitude',
                    'units': 'degrees_north',
                    'axis': 'Y'
                }
            },
            1: {
                'name': 'lon',
                'values': lon,
                'dtype': 'f',
                'atts': {
                    'long_name': 'Longitude',
                    'standard_name': 'longitude',
                    'units': 'degrees_east',
                    'axis': 'X'
                }
            }
        }

        variables = {
            0: {
                'name': 'vmax',
                'dims': ('lat', 'lon'),
                'values': speed,
                'dtype': 'f',
                'atts': {
                    'long_name': 'Maximum 3-second gust wind speed',
                    'standard_name': 'wind_speed_of_gust',
                    'units': 'm/s',
                    'actual_range': (np.min(speed), np.max(speed)),
                    'valid_range': (0.0, 200.),
                    'cell_methods': ('time: maximum '
                                     'time: maximum (interval: 3 seconds)'),
                    'grid_mapping': 'crs'
                }
            },
            1: {
                'name': 'ua',
                'dims': ('lat', 'lon'),
                'values': Vx,
                'dtype': 'f',
                'atts': {
                    'long_name': 'Eastward component of maximum wind speed',
                    'standard_name': 'eastward_wind',
                    'units': 'm/s',
                    'actual_range': (np.min(Vx), np.max(Vx)),
                    'valid_range': (-200., 200.),
                    'grid_mapping': 'crs'
                }
            },
            2: {
                'name': 'va',
                'dims': ('lat', 'lon'),
                'values': Vy,
                'dtype': 'f',
                'atts': {
                    'long_name': 'Northward component of maximim wind speed',
                    'standard_name': 'northward_wind',
                    'units': 'm/s',
                    'actual_range': (np.min(Vy), np.max(Vy)),
                    'valid_range': (-200., 200.),
                    'grid_mapping': 'crs'
                }
            },
            3: {
                'name': 'slp',
                'dims': ('lat', 'lon'),
                'values': P,
                'dtype': 'f',
                'atts': {
                    'long_name': 'Minimum air pressure at sea level',
                    'standard_name': 'air_pressure_at_sea_level',
                    'units': 'Pa',
                    'actual_range': (np.min(P), np.max(P)),
                    'valid_range': (70000., 115000.),
                    'cell_methods': 'time: minimum',
                    'grid_mapping': 'crs'
                }
            },
            4: {
                'name': 'crs',
                'dims': (),
                'values': None,
                'dtype': 'i',
                'atts': {
                    'grid_mapping_name': 'latitude_longitude',
                    'semi_major_axis': 6378137.0,
                    'inverse_flattening': 298.257222101,
                    'longitude_of_prime_meridian': 0.0
                }
            }
        }

        nctools.ncSaveGrid(filename, dimensions, variables, gatts=gatts)

    def saveWindToFile(self, trackfile, dset, filename):

        log.info("Writing wind to %s" % filename)

        trackfileDate = flModDate(trackfile)

        gatts = {
            'title': 'TCRM hazard simulation - synthetic event wind field',
            'tcrm_version': flProgramVersion(),
            'python_version': sys.version,
            'track_file': trackfile,
            'track_file_date': trackfileDate,
            'radial_profile': self.profileType,
            'boundary_layer': self.windFieldType,
            'beta': self.beta}

        # Add configuration settings to global attributes:
        for section in self.config.sections():
            for option in self.config.options(section):
                key = "{0}_{1}".format(section, option)
                value = self.config.get(section, option)
                gatts[key] = value

        dset.attrs.update(gatts)
        dset.to_netcdf(filename)


    def dumpGustsFromTrackfiles(self, trackfiles, windfieldPath,
                                timeStepCallback=None):
        """
        Helper method to dump the maximum wind speeds (gusts) observed over a
        region to netcdf files. One file is created for every track file.

        :type  trackfiles: list of str
        :param trackfiles: a list of track file filenames.

        :type  windfieldPath: str
        :param windfieldPath: the path where to store the gust output files.

        :type  filenameFormat: str
        :param filenameFormat: the format string for the output file names. The
                               default is set to 'gust-%02i-%04i.nc'.

        :type  progressCallback: function
        :param progressCallback: optional function to be called after a file is
                                 saved. This can be used to track progress.

        :type  timeStepCallBack: function
        :param timeStepCallback: optional function to be called at each
                                 timestep to extract point values for
                                 specified locations.

        """

        tracks = loadTracksFromFiles(sorted(trackfiles))

        self.dumpGustsFromTracks(tracks, windfieldPath,
                                 timeStepCallback=timeStepCallback)


def loadTracksFromFiles(trackfiles):
    """
    Generator that yields :class:`Track` objects from a list of track
    filenames.

    When run in parallel, the list `trackfiles` is distributed across the MPI
    processors using the `balanced` function. Track files are loaded in a lazy
    fashion to reduce memory consumption. The generator returns individual
    tracks (recall: a trackfile can contain multiple tracks) and only moves on
    to the next file once all the tracks from the current file have been
    returned.

    :type  trackfiles: list of strings
    :param trackfiles: list of track filenames. The filenames must include the
                       path to the file.
    """
    for f in balanced(trackfiles):
        msg = 'Calculating wind fields for tracks in %s' % f
        log.info(msg)
        tracks = loadTracks(f)
        for track in tracks:
            yield track


def loadTracks(trackfile):
    """
    Read tracks from a track .nc file and return a list of :class:`Track`
    objects.

    This calls the function `ncReadTrackData` to parse the track .nc
    file.

    :param str trackfile: the track data filename.

    :return: list of :class:`Track` objects.

    """

    tracks = ncReadTrackData(trackfile)
    return tracks


def create_nc(filename, lats, lons, trackfile, config, vars=['mslp','uwnd','vwnd'],
              descs=['Pressure','uwind','vwnd'],
              units=['hPa','ms-1','ms-1'],
              time_ref=datetime(1979,01,01)):

    trackfileDate = flModDate(trackfile)
    profileType = config.get('WindfieldInterface', 'profileType')
    windFieldType = config.get('WindfieldInterface', 'windFieldType')
    beta = config.getfloat('WindfieldInterface', 'beta')

    gatts = {
        'title': 'TCRM hazard simulation - synthetic event wind field',
        'tcrm_version': flProgramVersion(),
        'python_version': sys.version,
        'track_file': trackfile,
        'track_file_date': trackfileDate,
        'radial_profile': profileType,
        'boundary_layer': windFieldType,
        'beta': beta}

    # Add configuration settings to global attributes:
    for section in config.sections():
        for option in config.options(section):
            key = "{0}_{1}".format(section, option)
            value = config.get(section, option)
            gatts[key] = value

    log.info('Creating output file %s...'% filename)
    with Dataset(filename, 'w', format='NETCDF4') as nc:
        # Global attributes
        nc.Description = 'MetOcean Solutions Model Results'
        nc.history = 'Created ' + timemod.ctime(timemod.time())
        for key,val in gatts.items():
            setattr(nc,key,val)

        # Dimensions
        time = nc.createDimension('time', None)
        lat = nc.createDimension('lat', lats.size)
        lon = nc.createDimension('lon', lons.size)

        # Variables
        latnc = nc.createVariable('lat', 'f4', ('lat',))
        lonnc = nc.createVariable('lon', 'f4', ('lon',))
        timenc = nc.createVariable('time', 'i4', ('time',))
        timenc.units = time_ref.strftime('seconds since %Y-01-01T00:00:00 UTC')
        latnc[:] = lats
        lonnc[:] = lons
        ncvars={}
        for var in vars:
            ncvars[var] = nc.createVariable(var ,'f4', ('time','lat','lon'))

        for var, unit, desc, in zip(vars, units, descs,):
            setattr(ncvars[var], 'units', unit)
            setattr(ncvars[var], 'description', desc)

def write_record(filename, dt, record={}):
    log.debug('Writing record to file %s...'% filename)
    recdt = datetime(dt.year, dt.month, dt.day, dt.hour, dt.second)
    with Dataset(filename, 'a') as nc:
        times = nc.variables['time']
        dim=times.size
        rectime = date2num(recdt,units=times.units)
        for var, data in record.items():
            nc.variables[var][dim,:,:]=data

def loadTracksFromPath(path):
    """
    Helper function to obtain a generator that yields :class:`Track` objects
    from a directory containing track .csv files.

    This function calls `loadTracksFromFiles` to obtain the generator and track
    filenames are processed in alphabetical order.

    :type  path: str
    :param path: the directory path.
    """
    files = os.listdir(path)
    trackfiles = [pjoin(path, f) for f in files if f.startswith('tracks')]
    msg = 'Processing %d track files in %s' % (len(trackfiles), path)
    log.info(msg)
    return loadTracksFromFiles(sorted(trackfiles))

def append_record(ncfile,time,record_dict):
    data = xr.open_dataset(ncfile)
    rec = data.time.size + 1
    data[time][rec] = time
    for var, dat in record_dict.items():
        data[var]


def balanced(iterable):
    """
    Balance an iterator across processors.

    This partitions the work evenly across processors. However, it requires
    the iterator to have been generated on all processors before hand. This is
    only some magical slicing of the iterator, i.e., a poor man version of
    scattering.
    """
    P, p = pp.size(), pp.rank()
    return itertools.islice(iterable, p, None, P)


def run(configFile, callback=None):
    """
    Run the wind field calculations.

    :param str configFile: path to a configuration file.
    :param func callback: optional callback function to track progress.

    """

    log.info('Loading wind field calculation settings')

    # Get configuration

    config = ConfigParser()
    config.read(configFile)

    profileType = config.get('WindfieldInterface', 'profileType')
    windFieldType = config.get('WindfieldInterface', 'windFieldType')
    writeWinds = config.getboolean('WindfieldInterface', 'writeWinds')
    blendWinds = config.getboolean('WindfieldInterface', 'blendWinds')
    dtout = config.getfloat('WindfieldInterface', 'dtout')
    beta = config.getfloat('WindfieldInterface', 'beta')
    beta1 = config.getfloat('WindfieldInterface', 'beta1')
    beta2 = config.getfloat('WindfieldInterface', 'beta2')
    thetaMax = config.getfloat('WindfieldInterface', 'thetaMax')
    margin = config.getfloat('WindfieldInterface', 'Margin')
    resolution = config.getfloat('WindfieldInterface', 'Resolution')
    domain = config.get('WindfieldInterface', 'Domain')

    outputPath = config.get('Output', 'Path')
    windfieldPath = pjoin(outputPath, 'windfield')
    trackPath = pjoin(outputPath, 'tracks')

    gridLimit = None
    if config.has_option('Region','gridLimit'):
        gridLimit = config.geteval('Region', 'gridLimit')

    if config.has_option('WindfieldInterface', 'gridLimit'):
        gridLimit = config.geteval('WindfieldInterface', 'gridLimit')

    if config.has_section('Timeseries'):
        if config.has_option('Timeseries', 'Extract'):
            if config.getboolean('Timeseries', 'Extract'):
                from Utilities.timeseries import Timeseries
                log.debug("Timeseries data will be extracted")
                ts = Timeseries(configFile)
                timestepCallback = ts.extract
            else:
                def timestepCallback(*args):
                    """Dummy timestepCallback function"""
                    pass

    else:
        def timestepCallback(*args):
            """Dummy timestepCallback function"""
            pass

    thetaMax = math.radians(thetaMax)

    # Attempt to start the track generator in parallel
    global pp
    pp = attemptParallel()

    log.info('Running windfield generator')

    wfg = WindfieldGenerator(config=config,
                             margin=margin,
                             resolution=resolution,
                             profileType=profileType,
                             windFieldType=windFieldType,
                             beta=beta,
                             beta1=beta1,
                             beta2=beta2,
                             thetaMax=thetaMax,
                             gridLimit=gridLimit,
                             domain=domain)

    msg = 'Dumping gusts to %s' % windfieldPath
    log.info(msg)

    # Get the trackfile names and count

    files = os.listdir(trackPath)
    trackfiles = [pjoin(trackPath, f) for f in files if f.startswith('tracks')]
    nfiles = len(trackfiles)

    def progressCallback(i):
        """Define the callback function"""
        callback(i, nfiles)

    msg = 'Processing %d track files in %s' % (nfiles, trackPath)
    log.info(msg)

    # Do the work

    pp.barrier()

    wfg.dumpGustsFromTrackfiles(trackfiles, windfieldPath, timestepCallback)

    try:
        ts.shutdown()
    except NameError:
        pass

    pp.barrier()

    log.info('Completed windfield generator')
