import xarray as xr
import xesmf


PROJSTRING = "+proj=stere +lat_0=90 +lat_ts=71 +lon_0=-39 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

mardir = "/net2/ovs/MAR_Greenland/"
infile = f"{mardir}/MARv3.5.2-10km-monthly-ERA-Interim-1979.nc"
listvars = ["CC", "SP"]
fyear=1979
lyear=1980

# === Load Data

ds = xr.open_dataset(infile, decode_times=False)
ds = ds.rename({"LON": "lon", "LAT": "lat"})

# load MOM6 grid (make sure lon/lat are named lon/lat)
# replace with what you want:
om4grid = xr.open_dataset("/net2/rnd/CM4p25_grid.nc")

# === create the remapping weights

remap = xesmf.Regridder(ds, om4grid, "bilinear", periodic=False, unmapped_to_nan=True)

# === remap to MOM6 grid

for year in range(fyear, lyear+1):
    ds = xr.open_dataset(f"{mardir}/MARv3.5.2-10km-monthly-ERA-Interim-{year}.nc", decode_times=False)
    out = xr.Dataset()
    for var in listvars:
        out[var] = remap(ds[var])
    out.to_netcdf(f"MAR_greenland_remapped_{year}.nc")
