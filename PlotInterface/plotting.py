
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
        mpl = da.plot.contour(zorder=2, **kwargs)
        mpl = da.plot.contourf(zorder=1, **kwargs)
    else:
        raise Exception("Plot type %s not recognised" % ptype)
    coast = NaturalEarthFeature(category='physical', scale='50m',
                                facecolor='gray',
                                edgecolor='black',name='coastline')
    ax.add_feature(coast)
    plt.axes(ax)
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.xformatter = LONGITUDE_FORMATTER
    return mpl

