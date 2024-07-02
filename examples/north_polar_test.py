#!/usr/bin/env python

import numpy as np
from pyproj import CRS, Transformer
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

#pick a grid from https://nsidc.org/data/polar-stereo/ps_grids.html
projstring_np = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs "

crs = CRS.from_proj4(projstring_np)
proj = Transformer.from_crs(crs.geodetic_crs, crs)

# read in model lat/lon instead
lon = np.arange(0,360)
lat = np.arange(40,90)

# transform
lon2, lat2 = np.meshgrid(lon,lat)
x, y = proj.transform(lon2, lat2)


# angle of the projection wrt to EW
angle_X_NS = np.arctan2(y, x)
angle_X_EW = angle_X_NS - (np.pi / 2)
angle_EW_X = np.mod(- angle_X_EW + np.pi, 2*np.pi)

plt.figure(); plt.pcolormesh(x,y,angle_X_EW, cmap="hsv") ; plt.colorbar() ; plt.show()

# zonal wind test field
u = np.ones_like(x)
v = np.zeros_like(x)

def rotate_velocities(u, v, angle):
    """rotate velocities by angle (in radians)"""

    if angle.max() > 2 * np.pi:
        raise ValueError("angle must be in radians")

    u_r = u * np.cos(angle) + v * np.sin(angle)
    v_r = v * np.cos(angle) - u * np.sin(angle)

    return u_r, v_r


ur, vr = rotate_velocities(u,v, angle_EW_X)
plt.figure()
plt.quiver(x[::5,::5],y[::5,::5],ur[::5,::5],vr[::5,::5], angles = "xy")
plt.show()

