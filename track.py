import numpy as np
from scipy.io import loadmat


def load_track(filename="track.mat"):
    """
    Load track data from MATLAB file.

    Returns:
        inner_x, inner_y
        outer_x, outer_y
        center_x, center_y
    """

    data = loadmat(filename)

    track = data["track"][0, 0]

    inner = track["inner"]
    outer = track["outer"]
    center = track["center"]

    inner_x = inner[0, :]
    inner_y = inner[1, :]

    outer_x = outer[0, :]
    outer_y = outer[1, :]

    center_x = center[0, :]
    center_y = center[1, :]

    return (
        inner_x,
        inner_y,
        outer_x,
        outer_y,
        center_x,
        center_y
    )


def get_start_line(inner_x, inner_y, outer_x, outer_y):
    """
    Return the two endpoints of the start line.
    """

    start_inner = np.array([
        inner_x[0],
        inner_y[0]
    ])

    start_outer = np.array([
        outer_x[0],
        outer_y[0]
    ])

    return start_inner, start_outer