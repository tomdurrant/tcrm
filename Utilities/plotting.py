
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature import NaturalEarthFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def plotMap(da, ax=None, ptype='pcolormesh',cblabel=None,**kwargs):
    if not ax:
        ax=plt.axes(projection=ccrs.PlateCarree())
    kwargs.update({'ax': ax})
    if ptype=='pcolormesh':
        mpl = da.plot.pcolormesh(**kwargs)
    elif ptype=='contour':
        mpl = ax.contour(da.lon, da.lat, da,
                         transform=ccrs.PlateCarree(),zorder=2, **kwargs)
        mpl = ax.contourf(da.lon, da.lat, da,
                          transform=ccrs.PlateCarree(),zorder=1, **kwargs)
    else:
        raise Exception("Plot type %s not recognised" % ptype)
    coast = NaturalEarthFeature(category='physical', scale='50m',
                                facecolor='gray',
                                edgecolor='black',name='coastline')
    ax.add_feature(coast)
    plt.axes(ax)
    # cb = plt.colorbar(mpl, fraction=0.046, pad=0.04, orientation='vertical')
    # if cblabel:
    #     cb.set_label(cblabel)
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.xformatter = LONGITUDE_FORMATTER
    return ax

