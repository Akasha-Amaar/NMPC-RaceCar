import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.io import loadmat

from track import (
    load_track,
    get_start_line
)

from reference import (
    load_reference,
    compute_reference_heading,
    build_reference_states
)

from vehicle import bicycle_step

from visualization import draw_car

from mpc_track_obstacle import (
    optimize_controls,
    predict_trajectory
)


# ============================================================
# SETTINGS
# ============================================================

N = 25

NUM_LAPS = 1

MAX_SIM_STEPS = 1400

SEARCH_WINDOW = 60

LIVE_PAUSE = 0.01


# ============================================================
# OBSTACLE SETTINGS
# ============================================================

OBSTACLE_RADIUS = 0.06

SAFETY_MARGIN = 0.04

SAFE_RADIUS = (
    OBSTACLE_RADIUS
    +
    SAFETY_MARGIN
)


# ============================================================
# TRACK SETTINGS
# ============================================================

TRACK_MARGIN = 0.03


# ============================================================
# LAP SETTINGS
# ============================================================

POST_LINE_REFERENCE_POINTS = 2

ARM_AFTER_REFERENCE_POINTS = 50


# ============================================================
# 1. LOAD TRACK
# ============================================================

(
    inner_x,
    inner_y,
    outer_x,
    outer_y,
    center_x,
    center_y
) = load_track(
    "track.mat"
)


# ============================================================
# 2. LOAD REFERENCE
# ============================================================

ref_x, ref_y, sim_time = load_reference(
    "reference.mat"
)

ref_psi = compute_reference_heading(
    ref_x,
    ref_y
)

reference_states = build_reference_states(
    ref_x,
    ref_y,
    ref_psi,
    v_ref=0.3
)

NUM_REFERENCES = len(
    reference_states
)


# ============================================================
# 3. LOAD OBSTACLES
# ============================================================

obstacle_data = loadmat(
    "obstacles.mat"
)

obstacles = obstacle_data[
    "obstacles"
]


print(
    "Number of obstacles:",
    obstacles.shape[1]
)

print(
    "Obstacle radius:",
    OBSTACLE_RADIUS
)

print(
    "Safety margin:",
    SAFETY_MARGIN
)

print(
    "Safe radius:",
    SAFE_RADIUS
)


# ============================================================
# 4. START LINE
# ============================================================

start_inner, start_outer = get_start_line(
    inner_x,
    inner_y,
    outer_x,
    outer_y
)


# ============================================================
# 5. CALCULATE TRACK CORRIDOR
# ============================================================

def calculate_track_radii(
    reference_states,
    inner_x,
    inner_y,
    outer_x,
    outer_y,
    track_margin
):

    track_radii = []

    for ref_state in reference_states:

        p = ref_state[0]
        q = ref_state[1]


        # ----------------------------------------------------
        # Distance from reference point to inner boundary
        # ----------------------------------------------------

        inner_distances = np.sqrt(
            (inner_x - p) ** 2
            +
            (inner_y - q) ** 2
        )


        # ----------------------------------------------------
        # Distance from reference point to outer boundary
        # ----------------------------------------------------

        outer_distances = np.sqrt(
            (outer_x - p) ** 2
            +
            (outer_y - q) ** 2
        )


        nearest_inner = np.min(
            inner_distances
        )

        nearest_outer = np.min(
            outer_distances
        )


        half_width = min(
            nearest_inner,
            nearest_outer
        )


        allowed_radius = (
            half_width
            -
            track_margin
        )


        allowed_radius = max(
            allowed_radius,
            0.03
        )


        track_radii.append(
            allowed_radius
        )


    return np.array(
        track_radii
    )


track_radii = calculate_track_radii(
    reference_states,
    inner_x,
    inner_y,
    outer_x,
    outer_y,
    TRACK_MARGIN
)


print(
    "Minimum track corridor radius:",
    np.min(track_radii)
)

print(
    "Maximum track corridor radius:",
    np.max(track_radii)
)


# ============================================================
# 6. CYCLIC REFERENCE SEARCH
# ============================================================

def find_cyclic_reference(
    state,
    reference_states,
    current_index,
    search_window=60
):

    num_refs = len(
        reference_states
    )

    p = state[0]
    q = state[1]

    best_distance = float(
        "inf"
    )

    best_offset = 0


    for offset in range(
        search_window
    ):

        index = (
            current_index
            +
            offset
        ) % num_refs


        ref_p = reference_states[
            index,
            0
        ]

        ref_q = reference_states[
            index,
            1
        ]


        distance = np.sqrt(
            (ref_p - p) ** 2
            +
            (ref_q - q) ** 2
        )


        if distance < best_distance:

            best_distance = distance

            best_offset = offset


    nearest_index = (
        current_index
        +
        best_offset
    ) % num_refs


    return (
        nearest_index,
        best_offset
    )


# ============================================================
# 7. CYCLIC REFERENCE WINDOW
# ============================================================

def get_cyclic_reference_window(
    reference_states,
    track_radii,
    start_index,
    N
):

    num_refs = len(
        reference_states
    )


    indices = np.array([
        (
            start_index + k
        ) % num_refs

        for k in range(
            N + 1
        )
    ])


    reference_window = (
        reference_states[
            indices
        ]
    )


    radius_window = (
        track_radii[
            indices
        ]
    )


    return (
        reference_window,
        radius_window
    )


# ============================================================
# 8. START LINE CROSSING
# ============================================================

def crossed_start_line(
    previous_position,
    current_position,
    line_start,
    line_end
):

    def orientation(
        a,
        b,
        c
    ):

        value = (
            (b[1] - a[1])
            *
            (c[0] - b[0])
            -
            (b[0] - a[0])
            *
            (c[1] - b[1])
        )


        if abs(value) < 1e-10:

            return 0


        return (
            1
            if value > 0
            else 2
        )


    o1 = orientation(
        previous_position,
        current_position,
        line_start
    )

    o2 = orientation(
        previous_position,
        current_position,
        line_end
    )

    o3 = orientation(
        line_start,
        line_end,
        previous_position
    )

    o4 = orientation(
        line_start,
        line_end,
        current_position
    )


    return (
        o1 != o2
        and
        o3 != o4
    )


# ============================================================
# 9. INITIAL STATE
# ============================================================

state = reference_states[
    0
].copy()

state[4] = 0.0


# ============================================================
# 10. STORAGE
# ============================================================

state_history = [
    state.copy()
]

control_history = []

minimum_obstacle_distance_history = []

track_margin_history = []


# ============================================================
# 11. CONTROLLER VARIABLES
# ============================================================

ref_index = 0

previous_controls = None


# ============================================================
# 12. LAP VARIABLES
# ============================================================

lap_count = 0

total_reference_progress = 0

previous_position = (
    state[0:2].copy()
)

lap_detector_armed = False

waiting_after_line_crossing = False

progress_at_line_crossing = None

progress_at_lap_reset = 0


# ============================================================
# 13. LIVE PLOT SETUP
# ============================================================

plt.ion()

fig, ax = plt.subplots(
    figsize=(11, 8)
)

plt.show(
    block=False
)


# ============================================================
# 14. LIVE DRAW FUNCTION
# ============================================================

def draw_live_scene(
    step,
    state,
    predicted_states,
    ref_index,
    optimal_cost,
    minimum_obstacle_distance,
    remaining_track_margin
):

    ax.clear()


    # ========================================================
    # TRACK
    # ========================================================

    ax.plot(
        inner_x,
        inner_y,
        label="Inner boundary"
    )

    ax.plot(
        outer_x,
        outer_y,
        label="Outer boundary"
    )

    ax.plot(
        center_x,
        center_y,
        "--",
        label="Center line"
    )


    # ========================================================
    # REFERENCE
    # ========================================================

    ax.plot(
        ref_x,
        ref_y,
        ":",
        label="Reference"
    )


    # ========================================================
    # OBSTACLES
    # ========================================================

    for j in range(
        obstacles.shape[1]
    ):

        obstacle_circle = Circle(
            (
                obstacles[0, j],
                obstacles[1, j]
            ),
            OBSTACLE_RADIUS,
            alpha=0.65
        )

        ax.add_patch(
            obstacle_circle
        )


    # Dummy obstacle legend entry
    ax.plot(
        [],
        [],
        "o",
        markersize=8,
        alpha=0.65,
        label="Obstacles"
    )


    # ========================================================
    # START LINE
    # ========================================================

    ax.plot(
        [
            start_inner[0],
            start_outer[0]
        ],
        [
            start_inner[1],
            start_outer[1]
        ],
        linewidth=4,
        color="black",
        label="Start line"
    )


    # ========================================================
    # VEHICLE HISTORY
    # ========================================================

    history = np.array(
        state_history
    )


    ax.plot(
        history[:, 0],
        history[:, 1],
        linewidth=2,
        label="Vehicle path"
    )


    # ========================================================
    # CURRENT NMPC PREDICTION
    # ========================================================

    ax.plot(
        predicted_states[:, 0],
        predicted_states[:, 1],
        marker=".",
        linewidth=1.5,
        label="NMPC prediction"
    )


    # ========================================================
    # CURRENT REFERENCE POINT
    # ========================================================

    ax.scatter(
        reference_states[
            ref_index,
            0
        ],
        reference_states[
            ref_index,
            1
        ],
        s=45,
        label="Current reference"
    )


    # ========================================================
    # VEHICLE
    # ========================================================

    draw_car(
        ax,
        state
    )


    # ========================================================
    # LIVE INFORMATION
    # ========================================================

    current_lap = min(
        lap_count + 1,
        NUM_LAPS
    )


    ax.set_title(
        f"NMPC Live | "
        f"Lap {current_lap}/{NUM_LAPS} | "
        f"Step {step} | "
        f"v = {state[3]:.3f} m/s | "
        f"Cost = {optimal_cost:.2f}\n"
        f"Obstacle distance = "
        f"{minimum_obstacle_distance:.3f} m | "
        f"Track margin = "
        f"{remaining_track_margin:.3f} m"
    )


    ax.set_xlabel(
        "p [m]"
    )

    ax.set_ylabel(
        "q [m]"
    )

    ax.axis(
        "equal"
    )

    ax.grid(
        True
    )

    ax.legend(
        loc="upper right"
    )


    fig.canvas.draw_idle()

    fig.canvas.flush_events()

    plt.pause(
        LIVE_PAUSE
    )


# ============================================================
# 15. MAIN LIVE NMPC LOOP
# ============================================================

for step in range(
    MAX_SIM_STEPS
):


    # --------------------------------------------------------
    # Find current reference
    # --------------------------------------------------------

    (
        ref_index,
        forward_advance
    ) = find_cyclic_reference(
        state,
        reference_states,
        ref_index,
        search_window=SEARCH_WINDOW
    )


    total_reference_progress += (
        forward_advance
    )


    # --------------------------------------------------------
    # Build NMPC horizon
    # --------------------------------------------------------

    (
        reference_window,
        radius_window
    ) = get_cyclic_reference_window(
        reference_states,
        track_radii,
        ref_index,
        N
    )


    # --------------------------------------------------------
    # Solve NMPC
    # --------------------------------------------------------

    (
        optimal_controls,
        optimal_cost,
        optimizer_success,
        optimizer_message
    ) = optimize_controls(
        state,
        reference_window,
        radius_window,
        N,
        obstacles,
        OBSTACLE_RADIUS,
        SAFETY_MARGIN,
        initial_guess=previous_controls
    )


    if not optimizer_success:

        print(
            f"\nWARNING at step {step}: "
            f"{optimizer_message}"
        )


    # --------------------------------------------------------
    # Predicted trajectory
    # --------------------------------------------------------

    predicted_states = predict_trajectory(
        state,
        optimal_controls
    )


    # --------------------------------------------------------
    # First control
    # --------------------------------------------------------

    control = (
        optimal_controls[
            0
        ].copy()
    )


    control_history.append(
        control
    )


    # --------------------------------------------------------
    # Warm start
    # --------------------------------------------------------

    previous_controls = np.vstack([
        optimal_controls[1:],
        optimal_controls[-1]
    ])


    # --------------------------------------------------------
    # Move vehicle
    # --------------------------------------------------------

    state = bicycle_step(
        state,
        control
    )


    state_history.append(
        state.copy()
    )


    current_position = (
        state[0:2].copy()
    )


    # ========================================================
    # 16. OBSTACLE DISTANCE
    # ========================================================

    obstacle_distances = np.sqrt(
        (
            obstacles[0, :]
            -
            state[0]
        ) ** 2

        +

        (
            obstacles[1, :]
            -
            state[1]
        ) ** 2
    )


    minimum_obstacle_distance = np.min(
        obstacle_distances
    )


    minimum_obstacle_distance_history.append(
        minimum_obstacle_distance
    )


    # ========================================================
    # 17. TRACK MARGIN
    # ========================================================

    current_reference_position = (
        reference_states[
            ref_index,
            0:2
        ]
    )


    distance_from_reference = np.linalg.norm(
        state[0:2]
        -
        current_reference_position
    )


    remaining_track_margin = (
        track_radii[
            ref_index
        ]
        -
        distance_from_reference
    )


    track_margin_history.append(
        remaining_track_margin
    )


    # ========================================================
    # 18. LIVE DRAW
    # ========================================================

    draw_live_scene(
        step,
        state,
        predicted_states,
        ref_index,
        optimal_cost,
        minimum_obstacle_distance,
        remaining_track_margin
    )


    # ========================================================
    # 19. ARM LAP DETECTOR
    # ========================================================

    progress_since_reset = (
        total_reference_progress
        -
        progress_at_lap_reset
    )


    if (
        not lap_detector_armed
        and
        not waiting_after_line_crossing
        and
        progress_since_reset
        >=
        ARM_AFTER_REFERENCE_POINTS
    ):

        lap_detector_armed = True


    # ========================================================
    # 20. CHECK START LINE
    # ========================================================

    crossed_line = crossed_start_line(
        previous_position,
        current_position,
        start_inner,
        start_outer
    )


    if (
        crossed_line
        and
        lap_detector_armed
        and
        not waiting_after_line_crossing
    ):

        waiting_after_line_crossing = True

        lap_detector_armed = False

        progress_at_line_crossing = (
            total_reference_progress
        )


        print(
            "\nStart line crossed."
        )


    # ========================================================
    # 21. COMPLETE LAP
    # ========================================================

    if waiting_after_line_crossing:

        progress_after_crossing = (
            total_reference_progress
            -
            progress_at_line_crossing
        )


        if (
            progress_after_crossing
            >=
            POST_LINE_REFERENCE_POINTS
        ):

            lap_count += 1

            waiting_after_line_crossing = False

            progress_at_line_crossing = None

            progress_at_lap_reset = (
                total_reference_progress
            )


            print(
                "\n"
                "======================================"
            )

            print(
                f"        LAP {lap_count} COMPLETED"
            )

            print(
                "======================================\n"
            )


            if lap_count >= NUM_LAPS:

                break


    previous_position = (
        current_position.copy()
    )


# ============================================================
# 22. CONVERT HISTORY
# ============================================================

state_history = np.array(
    state_history
)

control_history = np.array(
    control_history
)

minimum_obstacle_distance_history = np.array(
    minimum_obstacle_distance_history
)

track_margin_history = np.array(
    track_margin_history
)


# ============================================================
# 23. STATE CHECK
# ============================================================

print(
    "\n--- State limits check ---"
)

print(
    "Max velocity:",
    np.max(
        state_history[:, 3]
    )
)

print(
    "Min velocity:",
    np.min(
        state_history[:, 3]
    )
)

print(
    "Max steering angle:",
    np.max(
        state_history[:, 4]
    )
)

print(
    "Min steering angle:",
    np.min(
        state_history[:, 4]
    )
)

print(
    "Steering limit:",
    np.pi / 6
)


# ============================================================
# 24. OBSTACLE CHECK
# ============================================================

print(
    "\n--- Obstacle check ---"
)

print(
    "Required safe radius:",
    SAFE_RADIUS
)

print(
    "Minimum actual obstacle distance:",
    np.min(
        minimum_obstacle_distance_history
    )
)


# ============================================================
# 25. TRACK CHECK
# ============================================================

print(
    "\n--- Track check ---"
)

print(
    "Minimum remaining corridor margin:",
    np.min(
        track_margin_history
    )
)


if np.min(
    track_margin_history
) >= -1e-3:

    print(
        "Track corridor respected."
    )

else:

    print(
        "WARNING: Track corridor violation."
    )


# ============================================================
# 26. LAP INFORMATION
# ============================================================

print(
    "\n--- Lap information ---"
)

print(
    "Completed laps:",
    lap_count
)

print(
    "Simulation steps:",
    len(state_history) - 1
)


# ============================================================
# 27. KEEP FINAL FIGURE OPEN
# ============================================================

plt.ioff()

plt.show()