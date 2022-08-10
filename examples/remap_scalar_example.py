import xarray as xr
from mom_in_stereo.generic_scalar import remap_scalar_to_MOM6


PROJSTRING = "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

infile = "/home/Olga.Sergienko/gridtopo_sandbox/Ant0_5/inv_MuBeta.nc"
MOMgrids = "/net2/rnd/Olga"

# === Load Data

ds = xr.open_dataset(infile)

# load MOM6 grid
hgrid = xr.open_dataset(f"{MOMgrids}/sosc73E82S_10rr.nc")
hgrid = hgrid.set_coords(["x", "y"])

# === Remap

remapped = remap_scalar_to_MOM6(ds, hgrid, projection=PROJSTRING)

# === Write to file

remapped.to_netcdf("inv_MuBeta_gridmom6.nc")
