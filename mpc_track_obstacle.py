import numpy as np
from scipy.optimize import minimize

from vehicle import bicycle_step


# ============================================================
# MPC WEIGHTS
# ============================================================

Q = np.diag([
    20.0,   # p
    20.0,   # q
    5.0,    # psi
    2.0,    # velocity
    1.0     # steering angle
])

R = np.diag([
    0.5,    # acceleration
    0.5     # steering rate
])

S = np.diag([
    50.0,
    50.0,
    10.0,
    5.0,
    1.0
])


# ============================================================
# ANGLE WRAPPING
# ============================================================

def wrap_angle(angle):

    return np.arctan2(
        np.sin(angle),
        np.cos(angle)
    )


# ============================================================
# TRACKING ERROR
# ============================================================

def tracking_error(
    state,
    reference_state
):

    error = (
        state
        -
        reference_state
    ).copy()

    error[2] = wrap_angle(
        error[2]
    )

    return error


# ============================================================
# STAGE COST
# ============================================================

def stage_cost(
    state,
    reference_state,
    control
):

    error = tracking_error(
        state,
        reference_state
    )

    state_cost = (
        error.T
        @ Q
        @ error
    )

    control_cost = (
        control.T
        @ R
        @ control
    )

    return (
        state_cost
        +
        control_cost
    )


# ============================================================
# PREDICT TRAJECTORY
# ============================================================

def predict_trajectory(
    initial_state,
    controls
):

    predicted_states = [
        initial_state.copy()
    ]

    state = initial_state.copy()

    for control in controls:

        state = bicycle_step(
            state,
            control
        )

        predicted_states.append(
            state.copy()
        )

    return np.array(
        predicted_states
    )


# ============================================================
# HORIZON COST
# ============================================================

def horizon_cost(
    initial_state,
    controls,
    reference_states
):

    predicted_states = predict_trajectory(
        initial_state,
        controls
    )

    total_cost = 0.0


    # --------------------------------------------------------
    # Stage costs
    # --------------------------------------------------------

    for k in range(
        len(controls)
    ):

        total_cost += stage_cost(
            predicted_states[k],
            reference_states[k],
            controls[k]
        )


    # --------------------------------------------------------
    # Terminal cost
    # --------------------------------------------------------

    terminal_error = tracking_error(
        predicted_states[-1],
        reference_states[-1]
    )

    total_cost += (
        terminal_error.T
        @ S
        @ terminal_error
    )

    return total_cost


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_controls(
    initial_state,
    reference_states,
    track_radii,
    N,
    obstacles,
    obstacle_radius,
    safety_margin,
    initial_guess=None
):


    # ========================================================
    # INITIAL GUESS
    # ========================================================

    if initial_guess is None:

        initial_guess = np.zeros(
            N * 2
        )

    else:

        initial_guess = (
            initial_guess.reshape(
                N * 2
            )
        )


    # ========================================================
    # CONTROL BOUNDS
    # ========================================================

    bounds = []

    for _ in range(N):

        bounds.append(
            (-0.6, 0.6)
        )

        bounds.append(
            (-1.5, 1.5)
        )


    # ========================================================
    # OBJECTIVE
    # ========================================================

    def objective(
        flat_controls
    ):

        controls = (
            flat_controls.reshape(
                N,
                2
            )
        )

        return horizon_cost(
            initial_state,
            controls,
            reference_states
        )


    # ========================================================
    # STATE CONSTRAINTS
    # ========================================================

    def state_constraints(
        flat_controls
    ):

        controls = (
            flat_controls.reshape(
                N,
                2
            )
        )

        predicted_states = (
            predict_trajectory(
                initial_state,
                controls
            )
        )

        v = predicted_states[
            :,
            3
        ]

        delta = predicted_states[
            :,
            4
        ]


        constraints = np.concatenate([
            0.7 - v,
            0.7 + v,

            np.pi / 6 - delta,
            np.pi / 6 + delta
        ])

        return constraints


    # ========================================================
    # OBSTACLE CONSTRAINTS
    # ========================================================

    def obstacle_constraints(
        flat_controls
    ):

        controls = (
            flat_controls.reshape(
                N,
                2
            )
        )

        predicted_states = (
            predict_trajectory(
                initial_state,
                controls
            )
        )


        safe_radius = (
            obstacle_radius
            +
            safety_margin
        )


        constraints = []


        # ----------------------------------------------------
        # Start from step 1.
        #
        # predicted_states[0] is the current real state,
        # which the optimizer cannot change.
        # ----------------------------------------------------

        for k in range(
            1,
            len(predicted_states)
        ):

            p = predicted_states[
                k,
                0
            ]

            q = predicted_states[
                k,
                1
            ]


            # ------------------------------------------------
            # Check every obstacle
            # ------------------------------------------------

            for j in range(
                obstacles.shape[1]
            ):

                p_obs = obstacles[
                    0,
                    j
                ]

                q_obs = obstacles[
                    1,
                    j
                ]


                # --------------------------------------------
                # Distance between vehicle center
                # and obstacle center
                # --------------------------------------------

                distance_squared = (
                    (p - p_obs) ** 2
                    +
                    (q - q_obs) ** 2
                )


                # --------------------------------------------
                # SLSQP requires:
                #
                # g(x) >= 0
                #
                # Therefore:
                #
                # distance² - safe_radius² >= 0
                # --------------------------------------------

                constraint_value = (
                    distance_squared
                    -
                    safe_radius ** 2
                )


                constraints.append(
                    constraint_value
                )


        return np.array(
            constraints
        )


    # ========================================================
    # TRACK CORRIDOR CONSTRAINTS
    # ========================================================

    def track_constraints(
        flat_controls
    ):

        controls = (
            flat_controls.reshape(
                N,
                2
            )
        )

        predicted_states = (
            predict_trajectory(
                initial_state,
                controls
            )
        )


        constraints = []


        # ----------------------------------------------------
        # Keep each predicted state inside the allowed
        # local corridor around the reference trajectory.
        # ----------------------------------------------------

        for k in range(
            1,
            len(predicted_states)
        ):

            p = predicted_states[
                k,
                0
            ]

            q = predicted_states[
                k,
                1
            ]


            ref_p = reference_states[
                k,
                0
            ]

            ref_q = reference_states[
                k,
                1
            ]


            allowed_radius = (
                track_radii[k]
            )


            distance_squared = (
                (p - ref_p) ** 2
                +
                (q - ref_q) ** 2
            )


            # --------------------------------------------
            # Inside corridor:
            #
            # allowed_radius² - distance² >= 0
            # --------------------------------------------

            constraint_value = (
                allowed_radius ** 2
                -
                distance_squared
            )


            constraints.append(
                constraint_value
            )


        return np.array(
            constraints
        )


    # ========================================================
    # ALL CONSTRAINTS
    # ========================================================

    constraints = [

        {
            "type": "ineq",
            "fun": state_constraints
        },

        {
            "type": "ineq",
            "fun": obstacle_constraints
        },

        {
            "type": "ineq",
            "fun": track_constraints
        }
    ]


    # ========================================================
    # SOLVE
    # ========================================================

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 120,
            "ftol": 1e-4,
            "disp": False
        }
    )


    optimal_controls = (
        result.x.reshape(
            N,
            2
        )
    )


    return (
        optimal_controls,
        result.fun,
        result.success,
        result.message
    )