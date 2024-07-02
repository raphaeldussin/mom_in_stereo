import numpy as np
from numba import njit
import xarray as xr


def fill(
    da,
    missing=1.0e36,
    max_iter=15000,
    threshold=1.0e-4,
    relax=0.6,
    periodic=True,
    initzonal=True,
):

    kwargs = dict(
        missing=missing,
        max_iter=max_iter,
        threshold=threshold,
        relax=relax,
        periodic=periodic,
        initzonal=initzonal,
    )

    return xr.apply_ufunc(
        fill_poisson,
        da,
        kwargs=kwargs,
        dask="parallelized",
        vectorize=True,
        input_core_dims=[("yh", "xh")],
        output_core_dims=[("yh", "xh")],
        output_dtypes=[da.dtype],
    )


@njit
def fill_poisson(
    npa,
    missing=1.0e36,
    max_iter=15000,
    threshold=1.0e-4,
    relax=0.6,
    periodic=True,
    initzonal=True,
):

    ny, nx = npa.shape
    a = npa.copy()
    sor = np.zeros((ny, nx))

    # init arrays
    for jj in range(ny):
        n_valid_pts = 0
        zonal_sum = 0.0

        for ji in range(nx):
            if a[jj, ji] == missing:
                sor[jj, ji] = relax
            else:
                n_valid_pts += 1
                zonal_sum += a[jj, ji]

        if n_valid_pts > 0:
            zonal_avg = zonal_sum / n_valid_pts
        else:
            zonal_avg = missing

        if initzonal:
            for ji in range(nx):
                if a[jj, ji] == missing:
                    a[jj, ji] = zonal_avg
        else:
            for ji in range(nx):
                if a[jj, ji] == missing:
                    a[jj, ji] = 0.0

    # iterate until convergence
    iters = 0
    max_residual = 1.0e+36

    while (iters < max_iter) or max_residual > threshold:
        max_residual = 0.0  # reinit
        iters += 1

        for jj in range(ny):
            jp1 = ny -1 if jj == ny else jj + 1
            #jp1 = ny if jj == ny else jj + 1
            jm1 = 2 if jj == 1 else jj - 1

            for ji in range(nx):

                if sor[jj, ji] != 0.0:
                    ip1 = ji + 1
                    im1 = ji - 1

                    if ji == 1:
                        im1 = nx if periodic else 2
                    if ji == nx:
                        ip1 = 1 if periodic else nx - 1

                    res = (
                        0.25 * (a[jj, im1] + a[jj, ip1] + a[jm1, ji] + a[jp1, ji])
                        - a[jj, ji]
                    )
                    res = res * sor[jj, ji]
                    a[jj, ji] += res
                    max_residual = max(abs(res), max_residual)

    return a
