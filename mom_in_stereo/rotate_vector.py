import numpy as np


def compute_angle_to_EW_stereo_grid(x, y):
    """compute the angle between local x-direction
    and E-W direction"""

    if len(x.shape) == 1:
        xx, yy = np.meshgrid(x, y)
    else:
        xx, yy = x, y

    # this is the angle between local x direction
    # and N-S direction (in radians)
    angle_X_NS = np.arctan2(yy, xx)
    # thus the angle wrt to E-W is
    angle_X_EW = angle_X_NS - (np.pi / 2)

    return angle_X_EW


def rotate_velocities(u, v, angle):
    """rotate velocities by angle (in radians)"""

    if angle.max() > 2 * np.pi:
        raise ValueError("angle must be in radians")

    u_r = u * np.cos(angle) + v * np.sin(angle)
    v_r = v * np.cos(angle) - u * np.sin(angle)

    return u_r, v_r


def rotate_EW_to_MOM6(uEW, vNS, angle_grid):
    """rotate from geographic axes to model axes"""
    # MOM6 angle_dx is in degrees, convert to rad
    angle_grid_rad = angle_grid * np.pi / 180.0
    return rotate_velocities(uEW, vNS, angle_grid_rad)


def rotate_MOM6_to_EW(u_model, v_model, angle_grid):
    """rotate from model axes to geographic axes"""
    # MOM6 angle_dx is in degrees, convert to rad
    angle_grid_rad = angle_grid * np.pi / 180.0
    return rotate_velocities(u_model, v_model, -1.0 * angle_grid_rad)


def rotate_stereo_to_EW(
    u_stereo,
    v_stereo,
    x,
    y,
):
    """rotate from stereo axes to geographic axes"""
    angle_X_EW = compute_angle_to_EW_stereo_grid(x, y)
    return rotate_velocities(u_stereo, v_stereo, angle_X_EW)


def rotate_EW_to_stereo(
    uEW,
    vNS,
    x,
    y,
):
    """rotate from stereo axes to geographic axes"""
    angle_X_EW = compute_angle_to_EW_stereo_grid(x, y)
    return rotate_velocities(uEW, vNS, -1.0 * angle_X_EW)
