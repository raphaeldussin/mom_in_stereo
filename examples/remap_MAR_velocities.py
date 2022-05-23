import xarray as xr
from mom_in_stereo.MAR import remap_MAR_velocities_to_MOM6
from mom_in_stereo.coords import add_lon_lat
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

MAR = "/net2/Olga.Sergienko/MARv3.6.4-Antarctica/ERA-Interim/"
MOMgrids = "/net2/rnd/Olga"

# === Load Data

# load MAR data
uvmar = xr.open_mfdataset(
    [
        f"{MAR}/MAR-ERA-Interim_UU_1979-2017_annual.nc4",
        f"{MAR}/MAR-ERA-Interim_VV_1979-2017_annual.nc4",
        f"{MAR}/MAR-ant35km-grid.nc",
    ],
    chunks={"time": 1},
)

# load MOM6 grid
hgrid = xr.open_dataset(f"{MOMgrids}/sosc73E82S_10rr.nc")

# === Remap velocities

uvmar_remapped = remap_MAR_velocities_to_MOM6(uvmar, hgrid)

# === Plot results

# create a coarser dataset
uvmar_coarse = (
    uvmar.isel(time=0, atmlay=0, x=slice(0, -1), y=slice(0, -1))
    .coarsen(x=10, y=10)
    .mean()
)
uvmar_remapped_coarse = (
    uvmar_remapped.isel(time=0, atmlay=0).coarsen(xh=10, yh=10).mean()
)


# quiver plot original data
plt.figure(figsize=[12, 8])
plt.contour(uvmar["x"], uvmar["y"], uvmar["GROUND"], [99.0], colors="k")
plt.quiver(uvmar_coarse["x"], uvmar_coarse["y"], uvmar_coarse["UU"], uvmar_coarse["VV"])


# quiver plot in MOM6 coords
plt.figure(figsize=[12, 8])
spproj = ccrs.SouthPolarStereo(central_longitude=0.0)
ax = plt.axes(projection=spproj)
uvmar_remapped_coarse.plot.quiver(
    "lon", "lat", "UU", "VV", ax=ax, transform=ccrs.PlateCarree()
)
ax.coastlines()
ax.set_extent([-180, 180, -50, -90], ccrs.PlateCarree())


plt.show()
