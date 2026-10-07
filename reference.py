import numpy as np
from scipy.io import loadmat


def load_reference(filename="reference.mat"):
    """
    Load reference trajectory from MATLAB file.
    """

    data = loadmat(filename)

    ref = data["Ref"]
    sim_time = int(data["simTime"][0, 0])

    ref_x = ref[0, :]
    ref_y = ref[1, :]

    return ref_x, ref_y, sim_time


def compute_reference_heading(ref_x, ref_y):
    """
    Calculate reference heading at every point.
    """

    ref_psi = []

    for k in range(len(ref_x) - 1):

        dx = ref_x[k + 1] - ref_x[k]
        dy = ref_y[k + 1] - ref_y[k]

        psi = np.arctan2(dy, dx)

        ref_psi.append(psi)

    # Use previous direction for final point
    ref_psi.append(ref_psi[-1])

    return np.array(ref_psi)


def build_reference_states(
    ref_x,
    ref_y,
    ref_psi,
    v_ref=0.3
):
    """
    Build complete reference states:

    [p_ref, q_ref, psi_ref, v_ref, delta_ref]
    """

    reference_states = []

    for k in range(len(ref_x)):

        state_ref = np.array([
            ref_x[k],
            ref_y[k],
            ref_psi[k],
            v_ref,
            0.0
        ])

        reference_states.append(state_ref)

    return np.array(reference_states)

def find_nearest_reference_index(
    state,
    reference_states,
    previous_index=0,
    search_window=60
):
    """
    Find the closest reference point ahead of the vehicle.

    Only search within a limited forward window so the controller
    does not jump to another nearby section of the track.
    """

    p = state[0]
    q = state[1]

    start_index = previous_index

    end_index = min(
        previous_index + search_window,
        len(reference_states)
    )

    positions = reference_states[
        start_index:end_index,
        0:2
    ]

    distances = np.sqrt(
        (positions[:, 0] - p) ** 2
        +
        (positions[:, 1] - q) ** 2
    )

    local_index = np.argmin(distances)

    nearest_index = start_index + local_index

    return nearest_index