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
    use_included_lonlat=False,
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
    use_included_lonlat: logical
        if True, use lon/lat included in dataset with names set
        by xgrid/ygrid to avoid recomputing arrays.
    """

    if use_included_lonlat:
        if len(ds[xgrid].values.shape) != 2:
            raise ValueError("included lon/lat are expected to be 2d arrays")
        # rename coords if needed
        if xgrid != "lon":
            ds["lon"] = xr.DataArray(data=ds[xgrid].values, dims=("y", "x"))
        if ygrid != "lat"
            ds["lat"] = xr.DataArray(data=ds[ygrid].values, dims=("y", "x"))
        # give proper units
        ds["lon"].attrs = dict(units="degrees_east")
        ds["lat"].attrs = dict(units="degrees_north")
    else:
        # create lon/lat arrays for MAR grid
        ds = add_lon_lat(ds, projection, x=xgrid, y=ygrid, units=units_grid)

    # create xESMF-friendly version of MOM6 grid
    ds_mom6 = MOM6_hgrid_to_xesmf(grid_mom6)

    # create the remapping weights
    remap = xesmf.Regridder(ds, ds_mom6, remap_method, periodic=False, unmapped_to_nan=True)

    # remap to MOM6 grid
    out = remap(ds)

    return out
