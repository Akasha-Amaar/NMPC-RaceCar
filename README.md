# Nonlinear Model Predictive Control for an Autonomous Race Car

This project implements a nonlinear Model Predictive Control (NMPC) framework for a simplified autonomous race car using a kinematic bicycle model.

The controller follows a predefined race-track reference trajectory while respecting vehicle constraints, avoiding obstacles, and remaining inside the allowed track corridor.

## Features

- Nonlinear kinematic bicycle model
- Euler discretization
- Reference trajectory tracking
- Nonlinear Model Predictive Control
- Warm-started optimization
- State constraints
  - Velocity limits
  - Steering-angle limits
- Input constraints
  - Acceleration limits
  - Steering-rate limits
- Predictive obstacle avoidance
- Track corridor constraints
- Automatic lap detection
- Live vehicle visualization
- NMPC prediction visualization
- Numerical verification of constraints

## Vehicle Model

The vehicle state is

\[
x =
\begin{bmatrix}
p & q & \psi & v & \delta
\end{bmatrix}^{T}
\]

where:

- \(p, q\): vehicle position
- \(\psi\): heading angle
- \(v\): velocity
- \(\delta\): steering angle

The control input is

\[
u =
\begin{bmatrix}
a & \omega
\end{bmatrix}^{T}
\]

where:

- \(a\): acceleration
- \(\omega\): steering rate

The nonlinear bicycle model is

\[
\dot p = v \cos(\psi + \beta)
\]

\[
\dot q = v \sin(\psi + \beta)
\]

\[
\dot \psi =
\frac{v}{l_r}\sin(\beta)
\]

\[
\dot v = a
\]

\[
\dot \delta = \omega
\]

with

\[
\beta =
\arctan
\left(
\frac{l_r}{l_f+l_r}
\tan(\delta)
\right)
\]

The continuous model is discretized using forward Euler integration.

## NMPC Formulation

At every simulation step, the controller predicts the future vehicle trajectory over a finite prediction horizon.

The optimization minimizes a tracking objective consisting of:

- position tracking error
- heading error
- velocity error
- steering-angle error
- control effort
- terminal-state error

Only the first optimized control input is applied to the vehicle.

The optimization is solved again at the next timestep using the shifted previous solution as a warm start.

## Constraints

The controller respects the following vehicle constraints:

### Control constraints

\[
|a| \leq 0.6
\]

\[
|\omega| \leq 1.5
\]

### State constraints

\[
|v| \leq 0.7
\]

\[
|\delta| \leq \frac{\pi}{6}
\]

## Obstacle Avoidance

Obstacle positions are loaded from `obstacles.mat`.

Each obstacle is represented as a circle.

For every predicted vehicle position and obstacle, the following constraint is enforced:

\[
(p-p_{obs})^2 +
(q-q_{obs})^2
\geq
r_{safe}^{2}
\]

where

\[
r_{safe} =
r_{obstacle}
+
r_{margin}
\]

The controller therefore begins changing the trajectory before reaching an obstacle because future states are already included in the optimization.

## Track Constraints

A local track corridor is calculated around the reference trajectory using the distance to the inner and outer track boundaries.

Each predicted state must remain within the allowed local corridor.

This prevents the optimizer from avoiding obstacles by leaving the race track.

## Lap Detection

A lap is counted when:

1. the vehicle has moved sufficiently far away from the starting area,
2. it physically crosses the start line,
3. it advances a small number of reference points after crossing.

The number of requested laps is configurable.

## Project Structure

```text
NMPC-RaceCar/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── vehicle.py
├── track.py
├── reference.py
├── visualization.py
├── mpc_track_obstacle.py
├── nmpc_track_obstacle_live.py
│
├── track.mat
├── reference.mat
└── obstacles.mat
```

## Main Files

### `vehicle.py`

Implements the nonlinear bicycle model and Euler integration.

### `track.py`

Loads the race-track geometry and start line.

### `reference.py`

Loads and constructs the reference trajectory used by the NMPC controller.

### `visualization.py`

Draws the four-wheel race car including steering-wheel orientation.

### `mpc_track_obstacle.py`

Contains the NMPC objective, trajectory prediction, state constraints, obstacle constraints, and track constraints.

### `nmpc_track_obstacle_live.py`

Runs the complete closed-loop simulation with live visualization and lap detection.

## Installation

Create a Python virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Simulation

Run:

```bash
python nmpc_track_obstacle_live.py
```

The simulation displays:

- race-track boundaries
- center line
- reference trajectory
- obstacles
- travelled vehicle path
- current NMPC prediction
- vehicle pose
- current lap
- velocity
- optimization cost
- obstacle distance
- remaining track margin

## Current Parameters

Typical working obstacle settings:

```python
OBSTACLE_RADIUS = 0.06
SAFETY_MARGIN = 0.04
```

The controller also uses:

```python
N = 25
```

for the NMPC prediction horizon.

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib
- Nonlinear Model Predictive Control
- Numerical Optimization
- Autonomous Vehicle Control

## Background

This project was developed as part of work in Advanced Predictive Control and extended into a modular Python implementation for autonomous race-track tracking and predictive obstacle avoidance.

## Future Improvements

Possible extensions include:

- exact vehicle-footprint collision checking
- improved track-boundary representation
- automatic obstacle-radius feasibility testing
- simulation recording and replay
- performance benchmarking
- alternative nonlinear optimizers
- reinforcement-learning comparison
