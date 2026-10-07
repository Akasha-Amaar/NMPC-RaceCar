import numpy as np


# -----------------------------------
# Vehicle parameters
# -----------------------------------

LF = 0.04
LR = 0.04
DT = 0.1


# -----------------------------------
# Initial vehicle state
# -----------------------------------

def initial_vehicle_state(center_x, center_y, start_index=0):
    """
    Create initial state:

    state = [p, q, psi, v, delta]
    """

    p = center_x[start_index]
    q = center_y[start_index]

    dx = center_x[start_index + 1] - center_x[start_index]
    dy = center_y[start_index + 1] - center_y[start_index]

    psi = np.arctan2(dy, dx)

    v = 0.0
    delta = 0.0

    return np.array([
        p,
        q,
        psi,
        v,
        delta
    ])


# -----------------------------------
# Continuous bicycle model
# -----------------------------------

def bicycle_dynamics(state, control):
    """
    Continuous-time bicycle model.

    state:
        [p, q, psi, v, delta]

    control:
        [a, omega]

    Returns:
        state derivative
        [p_dot, q_dot, psi_dot, v_dot, delta_dot]
    """

    p, q, psi, v, delta = state
    a, omega = control

    beta = np.arctan(
        (LR / (LF + LR)) * np.tan(delta)
    )

    p_dot = v * np.cos(psi + beta)

    q_dot = v * np.sin(psi + beta)

    psi_dot = (v / LR) * np.sin(beta)

    v_dot = a

    delta_dot = omega

    return np.array([
        p_dot,
        q_dot,
        psi_dot,
        v_dot,
        delta_dot
    ])


# -----------------------------------
# Discrete bicycle model
# -----------------------------------

def bicycle_step(state, control, dt=DT):
    """
    Euler discretization:

    x_next = x + dt * x_dot
    """

    state_dot = bicycle_dynamics(
        state,
        control
    )

    next_state = state + dt * state_dot

    return next_state