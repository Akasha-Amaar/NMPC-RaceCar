import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# Vehicle dimensions
W = 0.04          # car width
LF = 0.04         # distance from center to front axle
LR = 0.04         # distance from center to rear axle

# Wheel dimensions
WHEEL_L = 0.01
WHEEL_W = 0.005


def rotmat(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])


def transform_points(points, angle, translation):
    R = rotmat(angle)
    return R @ points + translation.reshape(2, 1)


def make_rectangle(length, width):
    return np.array([
        [ length/2,  length/2, -length/2, -length/2],
        [ width/2,  -width/2,  -width/2,   width/2]
    ])


def draw_car(ax, state):
    """
    state = [p, q, psi, v, delta]
    """
    p, q, psi, v, delta = state

    center = np.array([p, q])

    # -------------------------
    # Car body in local frame
    # x forward, y left/right
    # -------------------------
    body_local = np.array([
        [ LF,  LF, -LR, -LR],
        [ W/2, -W/2, -W/2,  W/2]
    ])

    body_global = transform_points(body_local, psi, center)

    # Orientation triangle
    tri_local = np.array([
        [ LF,      0.55*LF, 0.55*LF],
        [ 0.0,     0.18*W, -0.18*W]
    ])
    tri_global = transform_points(tri_local, psi, center)

    # -------------------------
    # Wheel shape in local wheel frame
    # -------------------------
    wheel_local = make_rectangle(WHEEL_L, WHEEL_W)

    # Wheel centers in vehicle local frame
    fl_center_local = np.array([ LF,  W/2])
    fr_center_local = np.array([ LF, -W/2])
    rl_center_local = np.array([-LR,  W/2])
    rr_center_local = np.array([-LR, -W/2])

    # Convert wheel centers to world frame
    Rpsi = rotmat(psi)

    fl_center = Rpsi @ fl_center_local + center
    fr_center = Rpsi @ fr_center_local + center
    rl_center = Rpsi @ rl_center_local + center
    rr_center = Rpsi @ rr_center_local + center

    # Front wheels rotate by steering angle delta
    fl_wheel = transform_points(wheel_local, psi + delta, fl_center)
    fr_wheel = transform_points(wheel_local, psi + delta, fr_center)

    # Rear wheels stay aligned with body
    rl_wheel = transform_points(wheel_local, psi, rl_center)
    rr_wheel = transform_points(wheel_local, psi, rr_center)

    # -------------------------
    # Draw
    # -------------------------
    body_patch = Polygon(body_global.T, closed=True, fill=False, edgecolor="black", linewidth=2)
    tri_patch = Polygon(tri_global.T, closed=True, fill=False, edgecolor="black", linewidth=2)

    fl_patch = Polygon(fl_wheel.T, closed=True, facecolor="black")
    fr_patch = Polygon(fr_wheel.T, closed=True, facecolor="black")
    rl_patch = Polygon(rl_wheel.T, closed=True, facecolor="black")
    rr_patch = Polygon(rr_wheel.T, closed=True, facecolor="black")

    ax.add_patch(body_patch)
    ax.add_patch(tri_patch)
    ax.add_patch(fl_patch)
    ax.add_patch(fr_patch)
    ax.add_patch(rl_patch)
    ax.add_patch(rr_patch)
    