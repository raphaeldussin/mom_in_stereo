from pyproj import CRS, Transformer
import numpy as np
import xarray as xr


def add_lon_lat(ds, PROJSTRING, x="x", y="y", units="m"):
    """add longitude and latitude as compute from the inverse projection
    given in PROJSTRING
    PARAMETERS:
    -----------
    ds: xarray.Dataset
    PROJSTRING: str
    """

    # create the coordinate reference system
    crs = CRS.from_proj4(PROJSTRING)
    # create the projection from lon/lat to x/y
    proj = Transformer.from_crs(crs.geodetic_crs, crs)
    xx, yy = np.meshgrid(ds.x.values, ds.y.values)
    # compute the lon/lat (x/y in km or in meters)
    if units == "m":
        lon, lat = proj.transform(xx, yy, direction="INVERSE")
    elif units == "km":
        lon, lat = proj.transform(1000 * xx, 1000 * yy, direction="INVERSE")
    else:
        raise ValueError("units must be in m or km")
    # add to dataset
    ds["lon"] = xr.DataArray(data=lon, dims=("y", "x"))
    ds["lat"] = xr.DataArray(data=lat, dims=("y", "x"))
    ds["lon"].attrs = dict(units="degrees_east")
    ds["lat"].attrs = dict(units="degrees_north")
    return ds


def MOM6_hgrid_to_xesmf(hgrid):
    """build xESMF-compat grid from MOM6 supergrid"""

    out = xr.Dataset()
    out["lon"] = xr.DataArray(hgrid["x"].values[1::2, 1::2], dims=("yh", "xh"))
    out["lat"] = xr.DataArray(hgrid["y"].values[1::2, 1::2], dims=("yh", "xh"))
    out["angle_dx"] = xr.DataArray(
        hgrid["angle_dx"].values[1::2, 1::2], dims=("yh", "xh")
    )

    out["lon_b"] = xr.DataArray(hgrid["x"].values[::2, ::2], dims=("yq", "xq"))
    out["lat_b"] = xr.DataArray(hgrid["y"].values[::2, ::2], dims=("yq", "xq"))

    return out
