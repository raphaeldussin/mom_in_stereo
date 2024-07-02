import xarray as xr
import xesmf
import numpy as np

from .coords import add_lon_lat, MOM6_hgrid_to_xesmf
from .rotate_vector import rotate_stereo_to_EW, rotate_EW_to_MOM6
from .extrap import fill

PROJSTRING_MAR = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def remap_MAR_velocities_to_MOM6(
    ds_MAR,
    grid_mom6,
    u="UU",
    v="VV",
    units_grid="km",
    xgrid="x",
    ygrid="y",
    remap_method="patch",
):
    """remap velocities from MAR to MOM6 grid

    Args:
    -----

    ds_MAR: xarray.Dataset
        dataset containing the MAR velocities and x/y
    grid_mom6: xarray.Dataset
        MOM6 supergrid (ocean_hgrid.nc)
    u: str
        name of u-velocity in MAR. Defaults to "UU"
    v: str
        name of v-velocity in MAR. Defaults to "VV"
    units_grid: str
        units of x/y in MAR grid (m/km). Defaults to km
    xgrid: str
        name of x-coord in MAR dataset. Defaults to "x"
    ygrid: str
        name of y-coord in MAR dataset. Defaults to "y"
    remap_method: str
        ESMF remapping scheme to use. Defaults to "patch"
    """

    # create lon/lat arrays for MAR grid
    ds_MAR = add_lon_lat(ds_MAR, PROJSTRING_MAR, x=xgrid, y=ygrid, units=units_grid)

    # create xESMF-friendly version of MOM6 grid
    ds_mom6 = MOM6_hgrid_to_xesmf(grid_mom6)

    # create the remapping weights
    remap = xesmf.Regridder(ds_MAR, ds_mom6, remap_method, periodic=False, unmapped_to_nan=True)

    # rotate MAR vectors to E-W/N-S, still on MAR grid (suffix)
    uEW_MAR, vNS_MAR = rotate_stereo_to_EW(
        ds_MAR[u], ds_MAR[v], ds_MAR[xgrid], ds_MAR[ygrid]
    )

    # remap to MOM6 grid
    uEW_MOM6 = remap(uEW_MAR)
    vNS_MOM6 = remap(vNS_MAR)

    # rotate to MOM6 grid
    u_MOM6, v_MOM6 = rotate_EW_to_MOM6(uEW_MOM6, vNS_MOM6, ds_mom6["angle_dx"])

    # extrapolate
    u_MOM6 = u_MOM6.fillna(1.0e36)
    v_MOM6 = v_MOM6.fillna(1.0e36)
    u_MOM6_extrap = fill(u_MOM6, initzonal=True)
    v_MOM6_extrap = fill(v_MOM6, initzonal=True)


    # pack all in a dataset
    out = xr.Dataset()
    out[u] = u_MOM6_extrap
    out[v] = v_MOM6_extrap

    return out
