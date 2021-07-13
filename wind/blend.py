#!/usr/bin/env python
# -*- coding: utf-8 -*-


from uds_client.query import Query
from uds_client.udscore import UDS
import tempfile
import urllib.request
from datetime import datetime, timedelta
import logging as log
log.level=20
import xarray as xr
import matplotlib.pyplot as plt
import os

"""
Blending tools needed to blend synthetic and background winds. It includes UDS data class
"""

class getData(object):

    def __init__(self, t0, t1,
                 var=['ugrd10m', 'vgrd10m', 'mslp'],
                 dset=['cfsr'],
                 bnd=[0,360,-80,80],
                 res=0.5,
                 dt=0.5,
                 udshost='http://uds1.rag.metocean.co.nz:9191/uds',
                 udsctls=None,
                 outnc = 'cfsr.nc',
                 udsconfig=None):
        self.t0 = t0
        self.t1 = t1
        self.var = var
        self.dset = dset
        self.bnd = bnd
        self.res = res
        self.udshost = udshost
        self.udsctls = udsctls
        self.udsconfig = udsconfig
        self.dt=dt
        self.outnc = outnc
        self.nx = (bnd[1] - bnd[0])/res + 1
        self.ny = (bnd[3] - bnd[2])/res + 1
        # Put standard variables in qdict
        if self.udsconfig:
            self.udsctls = self.get_data_config()
            self.udshost = None
        if self.udsctls:
            self.udshost = None
        self.qdict = {
            'fmt': ['nc'],
            'dim': None,
            'type':['fc','hc'],
            'stepback': [0],
            'dt': self.dt,
            }
        self.qdict.update({'var': list(self.var)})
        self.qdict.update({'dset': self.dset})
        self.qdict.update({'bnd': self.bnd})
        self.qdict.update({'dim': [self.nx,self.ny]})
        self.getUDS()

    def get_data_config(self):
        with open(self.udsconfig) as udsconfig:
            xmlfiles = []
            for line in udsconfig.readlines():
                if re.match('control-files', line):
                    ctlfiles = line.strip().split('=')[-1].split(',')
                    for ctlf in ctlfiles:
                        ctlf = ctlf.strip()
                        ctlf = os.path.join(self.udsrootdir, ctlf) if re.match(
                            "^etc", ctlf) else ctlf
                        files = glob.glob(ctlf)
                        xmlfiles += files
                        break
        return xmlfiles

    def getUDS(self,):
        self.qdict.update({'time': [self.t0.strftime('%Y%m%d.%H%M%S'),
                           (self.t1+timedelta(hours=self.dt)).strftime('%Y%m%d.%H%M%S')]})
        log.debug("Downloading %s data from the UDS for %s to %s" %
                 (self.qdict['dset'],self.qdict['time'][0],self.qdict['time'][1]))
        query = Query(self.qdict)
        log.debug("Query %s" % query)
        #outnc = tempfile.mktemp()
        #outnc = str("uds-%s-%s.nc" % (self.qdict['time'][0],self.qdict['time'][1]))
        if os.path.isfile(self.outnc):
            log.info("Using already downloaded file %s" % self.outnc)
            self.filename = self.outnc
        else:
            if self.udshost:
                nc = None
                url = self.udshost + '?' + query.str()
                log.info('Requesting url ' + url)
                log.debug('Saving file to %s' % self.outnc)
                self.filename, headers = urllib.request.urlretrieve(url, self.outnc)
            elif self.udsctls:
                uds = UDS(ctlfile=self.udsctls)
                uds.logger = log
                uds.getDSList(query)
                uds.getData(query, self.outnc)
                log.debug('Saving file to %s' % self.outnc)
                self.filename = self.outnc
        self.dset = xr.open_dataset(self.filename)

def blendWeights(R, sig, mu=0):
    # this seems to be the used one, not the one in windmodels
    """
    The weights for blending of wind field.
    R   :: Radial distance from centre of cycle
    sig :: Standard deviation of guassion distribution
    mu  :: Mean of guassion distribution
    """
    from Utilities.distributions import gauss
    return gauss(R,sig,mu)


def test():
    t0 = datetime(2000,1,2)
    t1 = datetime(2000,1,2,6)
    #data = getData(t0,udsctls=['/source/hindcast-config/uds/etc/uds.cfsr.xml'])
    data = getData(t0,t1)
    data.dset.mslp[0].plot()
    plt.show()

if __name__ == "__main__":
    test()

