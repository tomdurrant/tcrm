import os
import sys
import cPickle
import NumpyTestCase
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import math

from os.path import join as pjoin, realpath, isdir, dirname

from Utilities.config import ConfigParser
from tcevent import doOutputDirectoryCreation
from Evaluate import interpolateTracks
from wind import WindfieldGenerator, loadTracksFromFiles, loadTracks

from wind.windmodels import *
from plotting import plotMap
from Utilities.progressbar import SimpleProgressBar as ProgressBar
from Utilities.parallel import attemptParallel

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


    thetaMax = math.radians(thetaMax)

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

    msg = 'Processing %d track files in %s' % (nfiles, trackPath)
    log.info(msg)

    for trackfile in trackfiles:
        gust, bearing, Vx, Vy, P, lon, lat, dset = wfg.calculateExtremesFromTrackfile(trackfile, callback=None)

    projection=ccrs.PlateCarree()
    plot_set(dset,'mslp',2)
    plt.show()


def plot_set(dset,var,timestep,projection=ccrs.PlateCarree()):
    fig, axs = plt.subplots(2, 2, figsize=(15,10),
                           subplot_kw={'projection':projection})
    vmax=dset[var].max()
    vmin=dset[var].min()
    plotMap(dset[var+'_tc'][timestep],ax=axs[0,0],vmin=vmin,vmax=vmax); plt.title('Parametric')
    plotMap(dset[var+'_bg'][timestep],ax=axs[1,0],vmin=vmin,vmax=vmax); plt.title('Background')
    plotMap(dset['bweights'][timestep],ax=axs[0,1]); plt.title('TC Weighting')
    plotMap(dset[var][timestep],ax=axs[1,1],vmin=vmin,vmax=vmax); plt.title('Blended')


def main(configFile):
    """
    Main function to execute the :mod:`wind`.

    :param str configFile: Path to configuration file.

    """
    config = ConfigParser()
    config.read(configFile)
    doOutputDirectoryCreation(configFile)

    trackFile = config.get('DataProcess', 'InputFile')
    source = config.get('DataProcess', 'Source')
    delta = 1/2.
    outputPath = pjoin(config.get('Output','Path'), 'tracks')
    outputTrackFile = pjoin(outputPath, "tracks.interp.nc")

    # This will save interpolated track data in TCRM format:
    interpTrack = interpolateTracks.parseTracks(configFile, trackFile,
                                                source, delta,
                                                outputTrackFile,
                                                interpolation_type='akima')

    showProgressBar = config.get('Logging', 'ProgressBar')

    pbar = ProgressBar('Calculating wind fields: ', showProgressBar)

    def status(done, total):
        pbar.update(float(done)/total)

    run(configFile, status)

    # doWindfieldPlotting(configFile)
    # doTimeseriesPlotting(configFile)

if __name__ == "__main__":
    configFile = './test.ini'
    main(configFile)

