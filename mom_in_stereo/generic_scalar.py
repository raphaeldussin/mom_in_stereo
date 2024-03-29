import xarray as xr
import xesmf

from .coords import add_lon_lat, MOM6_hgrid_to_xesmf

PROJSTRING = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"


def remap_scalar_to_MOM6(
    ds,
    grid_mom6,
    units_grid="m",
    xgrid="x",
    ygrid="y",
    remap_method="bilinear",
    projection=PROJSTRING,
):
    """remap scalar from stereographic to MOM6 grid

    Args:
    -----

    ds: xarray.Dataset
        dataset containing the scalar field and x/y
    grid_mom6: xarray.Dataset
        MOM6 supergrid (ocean_hgrid.nc)
    units_grid: str
        units of x/y in MAR grid (m/km). Defaults to km
    xgrid: str
        name of x-coord in MAR dataset. Defaults to "x"
    ygrid: str
        name of y-coord in MAR dataset. Defaults to "y"
    remap_method: str
        ESMF remapping scheme to use. Defaults to "bilinear"
    projection: str
        PROJSTRING to use for input stereographic grid
    """

    # create lon/lat arrays for MAR grid
    ds = add_lon_lat(ds, projection, x=xgrid, y=ygrid, units=units_grid)

    # create xESMF-friendly version of MOM6 grid
    ds_mom6 = MOM6_hgrid_to_xesmf(grid_mom6)

    # create the remapping weights
    remap = xesmf.Regridder(ds, ds_mom6, remap_method, periodic=False, unmapped_to_nan=True)

    # remap to MOM6 grid
    out = remap(ds)

    return out
