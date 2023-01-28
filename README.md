# RL Missile Sim: Planar Interception & Guidance Simulator

`rl_missile_sim` is an open-source, high-fidelity planar homing missile guidance simulator and reinforcement learning testbed. Implemented in Python as a modern [Gymnasium](https://gymnasium.farama.org/) environment (`rl_missile_sim.envs.MissileSim`), it bridges classical aerospace Guidance, Navigation, and Control (GNC) theory with state-of-the-art Deep Reinforcement Learning (DRL) algorithms via [Stable-Baselines3](https://stable-baselines3.readthedocs.io/).

The environment models 2D atmospheric kinematic engagements between a high-speed pursuer (missile) and a maneuvering or non-maneuvering target (UAV) in an obstacle-strewn tactical airspace. It includes built-in implementations of classical Proportional Navigation (ProNav), Linear Quadratic Regulator (LQR) terminal guidance, Zero-Effort-Miss (ZEM) Optimal Guidance, and Proximal Policy Optimization (PPO) neural policies.

---

## Table of Contents

- [RL Missile Sim: Planar Interception \& Guidance Simulator](#rl-missile-sim-planar-interception--guidance-simulator)
  - [Table of Contents](#table-of-contents)
  - [Coordinate Systems and Kinematics](#coordinate-systems-and-kinematics)
    - [Planar Point-Mass Kinematics](#planar-point-mass-kinematics)
    - [Discrete Integration](#discrete-integration)
  - [Missile and Target State Representation](#missile-and-target-state-representation)
  - [Relative Engagement Geometry](#relative-engagement-geometry)
  - [Observation and Action Spaces](#observation-and-action-spaces)
    - [Observation Space Vector ($\\mathbb{R}^{33}$)](#observation-space-vector-mathbbr33)
    - [Forward-Looking LIDAR Subsystem](#forward-looking-lidar-subsystem)
    - [Action Space](#action-space)
  - [Guidance Policy Derivations](#guidance-policy-derivations)
    - [1. Proportional Navigation (ProNav / PN)](#1-proportional-navigation-pronav--pn)
      - [Mathematical Derivation](#mathematical-derivation)
      - [Collision Triangle and Stability](#collision-triangle-and-stability)
    - [2. Linear Quadratic Regulator (LQR) Guidance](#2-linear-quadratic-regulator-lqr-guidance)
      - [State-Space Double-Integrator Formulation](#state-space-double-integrator-formulation)
      - [Cost Functional and Riccati Equation](#cost-functional-and-riccati-equation)
      - [Normal Acceleration Projection](#normal-acceleration-projection)
    - [3. Optimal Guidance Law (OGL / Zero Effort Miss)](#3-optimal-guidance-law-ogl--zero-effort-miss)
      - [Concept of Zero-Effort-Miss (ZEM)](#concept-of-zero-effort-miss-zem)
      - [Optimal Control Derivation via Pontryagin's Principle](#optimal-control-derivation-via-pontryagins-principle)
    - [4. Deep Reinforcement Learning (PPO Policy)](#4-deep-reinforcement-learning-ppo-policy)
      - [Multi-Objective Reward Function](#multi-objective-reward-function)
      - [Proximal Policy Optimization (PPO) Formulation](#proximal-policy-optimization-ppo-formulation)
      - [Neural Architecture Hyperparameters](#neural-architecture-hyperparameters)
  - [Environment Dynamics, Obstacles, and Rewards](#environment-dynamics-obstacles-and-rewards)
    - [Engagement Limits and Truncation](#engagement-limits-and-truncation)
    - [Obstacle Field Layout](#obstacle-field-layout)
  - [Installation and Quickstart](#installation-and-quickstart)
    - [Prerequisites](#prerequisites)
    - [1. Installation via uv](#1-installation-via-uv)
    - [2. CUDA GPU Acceleration on Windows](#2-cuda-gpu-acceleration-on-windows)
  - [CLI Reference](#cli-reference)
    - [Policy Evaluation](#policy-evaluation)
    - [Training PPO Agents](#training-ppo-agents)
    - [Interactive Pygame Demonstration](#interactive-pygame-demonstration)
    - [Running Unit Tests](#running-unit-tests)
  - [Project Architecture](#project-architecture)
  - [References](#references)

---

## Coordinate Systems and Kinematics

The simulation is formulated in a planar North-East (NE) inertial reference frame $\mathcal{F}_I = \{\hat{\mathbf{i}}_N, \hat{\mathbf{i}}_E\}$ centered at the midpoint of a $20\,\text{km} \times 20\,\text{km}$ tactical engagement area $[-R_{\max}/2, \, R_{\max}/2]^2$, where $R_{\max} = 20{,}000\,\text{m}$.

```
             North (+N, y-axis)
                     ^
                     |      /  v_M (Heading psi_M)
                     |     /
                     |    /
                     |   * Missile [N_M, E_M]
                     |    \
                     |     \   LOS vector r
                     |      \
                     |       * Target [N_T, E_T]
                     +----------------------------> East (+E, x-axis)
                   (0,0)
```

### Planar Point-Mass Kinematics

Both entities (pursuer and target) are governed by point-mass non-holonomic equations of motion with constant forward velocity magnitude $V$ and heading angle $\psi \in [-\pi, \pi]$ measured clockwise from the North axis towards East:

$$\mathbf{p} = \begin{bmatrix} N \\ E \end{bmatrix}, \quad \mathbf{v} = \begin{bmatrix} N_v \\ E_v \end{bmatrix} = \begin{bmatrix} V \cos\psi \\ V \sin\psi \end{bmatrix}$$

$$\psi = \text{atan2}(E_v, N_v)$$

The missile control command is a scalar normal acceleration $a_n \in [-a_{\max}, a_{\max}]$ applied orthogonally to the instantaneous velocity vector $\mathbf{v}_M$. The continuous-time equations of motion are:

$$\begin{aligned}
\dot{N} &= V \cos\psi \\
\dot{E} &= V \sin\psi \\
\dot{\psi} &= \frac{a_n}{V}
\end{aligned}$$

Because the control authority acts strictly perpendicular to the trajectory ($\mathbf{a}_n \perp \mathbf{v}$), no work is done on the kinetic energy of the vehicle, preserving forward speed:

$$\frac{d}{dt}\left(\frac{1}{2}\|\mathbf{v}\|^2\right) = \mathbf{v} \cdot \mathbf{a}_n = 0 \implies \|\mathbf{v}\| = V = \text{constant}$$

### Discrete Integration

The simulation advances at a fixed timestep $\Delta t = 0.1\,\text{s}$ via forward Euler integration:

$$\begin{aligned}
\psi_{k+1} &= \text{wrap}_{[-\pi, \pi]}\left(\psi_k + \frac{a_{n, k}}{V_M} \Delta t\right) \\
N_{v, k+1} &= V_M \cos\psi_{k+1} \\
E_{v, k+1} &= V_M \sin\psi_{k+1} \\
N_{k+1} &= N_k + N_{v, k+1} \Delta t \\
E_{k+1} &= E_k + E_{v, k+1} \Delta t
\end{aligned}$$

Positions are bounded to the engagement corridor $[-10{,}000\,\text{m}, +10{,}000\,\text{m}]$ along both axes.

---

## Missile and Target State Representation

Each vehicle is instantiated through the `Entity` class:

$$\mathbf{x}_{\text{entity}} = \begin{bmatrix} N & E & \psi & N_v & E_v & r_{\text{rad}} \end{bmatrix}^T$$

- **Pursuer (Missile)**:
  - Initial speed: $V_M \in [0.7 V_{\max}, V_{\max}]$, with $V_{\max} = 1{,}700\,\text{m/s}$ (Mach $\sim 5$)
  - Heading: $\psi_M \in [-\gamma_{\min}, \gamma_{\max}]$ with $\gamma_{\min} = 10^\circ, \gamma_{\max} = 80^\circ$
  - Physical collision radius: $r_{\text{rad}} = 0.19\,\text{m}$ (representing an air-to-air interceptor such as the R-37)
  - Control authority: $a_{\max} = 200 g \approx 1{,}962\,\text{m/s}^2$
- **Target (UAV)**:
  - Initial speed: $V_T \in [80\,\text{m/s}, 120\,\text{m/s}]$
  - Heading: $\psi_T \in [-\gamma_{\min}, \gamma_{\max}]$
  - Physical collision radius: $r_{\text{rad}} = 2.0\,\text{m}$ (representative of an MQ-9 Reaper class UAV)
  - Non-maneuvering baseline ($\dot{\psi}_T = 0$)

---

## Relative Engagement Geometry

Let the relative position vector from missile to target be:

$$\mathbf{r} = \mathbf{p}_T - \mathbf{p}_M = \begin{bmatrix} N_T - N_M \\ E_T - E_M \end{bmatrix} = \begin{bmatrix} R_N \\ R_E \end{bmatrix}$$

The relative velocity vector is:

$$\mathbf{v}_{\text{rel}} = \mathbf{v}_T - \mathbf{v}_M = \begin{bmatrix} N_{v, T} - N_{v, M} \\ E_{v, T} - E_{v, M} \end{bmatrix} = \begin{bmatrix} \Delta v_N \\ \Delta v_E \end{bmatrix}$$

```
                E (East)
                ^
                |                      * Target (p_T, v_T)
                |                     /
                |                    /
                |                   /   Line-Of-Sight (LOS)
                |        r         /
                |                 /
                |                /  theta (inertial LOS angle)
                |               /--------->
                |   *          /
                |  /  Missile (p_M, v_M)
                | / psi_M
                |/
                +----------------------------------------> N (North)
```

From these vectors, the scalar engagement variables are derived:

1. **Slant Range ($r$)**:
   $$r = \|\mathbf{r}\|_2 = \sqrt{R_N^2 + R_E^2}$$

2. **LOS Unit Vector ($\hat{\mathbf{r}}$)**:
   $$\hat{\mathbf{r}} = \frac{\mathbf{r}}{r} = \begin{bmatrix} \cos\theta \\ \sin\theta \end{bmatrix}$$
   where $\theta = \text{atan2}(R_E, R_N)$ is the inertial Line-Of-Sight angle.

3. **Range Rate ($\dot{r}$)**:
   $$\dot{r} = \frac{d}{dt}\left(\sqrt{\mathbf{r} \cdot \mathbf{r}}\right) = \frac{\mathbf{r} \cdot \mathbf{v}_{\text{rel}}}{r} = \frac{R_N \Delta v_N + R_E \Delta v_E}{r}$$

4. **Closing Velocity ($V_c$)**:
   The rate of range decrease:
   $$V_c = -\dot{r} = -\frac{\mathbf{r} \cdot \mathbf{v}_{\text{rel}}}{r}$$

5. **Body-Relative Line-Of-Sight Angle ($\lambda$)**:
   $$\lambda = \text{wrap}_{[-\pi, \pi]}\Big(\text{atan2}(R_E, R_N) - \psi_M\Big)$$

6. **Line-Of-Sight (LOS) Rate ($\dot{\lambda} \equiv \dot{\theta}$)**:
   Differentiating the inertial LOS angle $\theta = \arctan(R_E / R_N)$ with respect to time yields:
   $$\dot{\theta} = \frac{R_N \dot{R}_E - R_E \dot{R}_N}{R_N^2 + R_E^2} = \frac{R_N \Delta v_E - R_E \Delta v_N}{r^2} = \frac{\mathbf{r} \times \mathbf{v}_{\text{rel}}}{r^2}$$
   In the simulation, this angular rate is wrapped to $[-\pi, \pi]\,\text{rad/s}$.

---

## Observation and Action Spaces

### Observation Space Vector ($\mathbb{R}^{33}$)

The observation vector provided to guidance policies contains $17$ kinematic features plus $16$ forward-looking LIDAR range measurements:

| Index Range | Symbol | Dimension | Description | Typical Bounds |
|:---:|:---:|:---:|:---|:---:|
| `0:2` | $N_M, E_M$ | 2 | Missile North, East inertial positions | $[-10\,\text{km}, 10\,\text{km}]$ |
| `2` | $\psi_M$ | 1 | Missile heading angle | $[-\pi, \pi]\,\text{rad}$ |
| `3:5` | $N_{v, M}, E_{v, M}$ | 2 | Missile velocity components | $[-1{,}700, 1{,}700]\,\text{m/s}$ |
| `5` | $r_{\text{rad}, M}$ | 1 | Missile lethal/bounding radius | $0.19\,\text{m}$ |
| `6:8` | $N_T, E_T$ | 2 | Target North, East inertial positions | $[-10\,\text{km}, 10\,\text{km}]$ |
| `8` | $\psi_T$ | 1 | Target heading angle | $[-\pi, \pi]\,\text{rad}$ |
| `9:11` | $N_{v, T}, E_{v, T}$ | 2 | Target velocity components | $[-120, 120]\,\text{m/s}$ |
| `11` | $r_{\text{rad}, T}$ | 1 | Target bounding radius | $2.0\,\text{m}$ |
| `12` | $V_M$ | 1 | Missile scalar speed $\|\mathbf{v}_M\|$ | $[1{,}190, 1{,}700]\,\text{m/s}$ |
| `13` | $r$ | 1 | Instantaneous slant range | $[0, 28{,}284]\,\text{m}$ |
| `14` | $V_c$ | 1 | Closing velocity ($-\dot{r}$) | $[-2{,}000, 2{,}000]\,\text{m/s}$ |
| `15` | $\lambda$ | 1 | Body-relative LOS angle | $[-\pi, \pi]\,\text{rad}$ |
| `16` | $\dot{\lambda}$ | 1 | Line-of-sight angular rate | $[-\pi, \pi]\,\text{rad/s}$ |
| `17:33` | $\mathbf{d}_{\text{lidar}}$ | 16 | Forward-looking LIDAR rangefinder beams | $[0, 10{,}000]\,\text{m}$ |

### Forward-Looking LIDAR Subsystem

The pursuer carries a $16$-beam planar rangefinder scanning a Field-of-View (FOV) of $\pi\,\text{rad}$ ($180^\circ$) symmetric about the missile's current heading:

$$\theta_i = \psi_M - \frac{\text{FOV}}{2} + \left(\frac{i}{N_{\text{beams}} - 1}\right)\text{FOV}, \quad i \in \{0, \dots, 15\}$$

Each ray is projected against circular obstacles centered at $\mathbf{p}_{o, j} = [N_{o, j}, E_{o, j}]^T$ with radius $R_{o, j}$. Solving the ray-sphere quadratic intersection:

$$\|\mathbf{p}_M + s \hat{\mathbf{u}}_i - \mathbf{p}_{o, j}\|^2 = R_{o, j}^2 \implies s^2 - 2 (\mathbf{p}_{o, j} - \mathbf{p}_M) \cdot \hat{\mathbf{u}}_i \, s + \|\mathbf{p}_{o, j} - \mathbf{p}_M\|^2 - R_{o, j}^2 = 0$$

returns the shortest positive real distance $s \in (0, R_{\text{lidar}}]$.

### Action Space

The action space is a continuous 1D Box:

$$a_n \in [-a_{\max}, +a_{\max}] \subset \mathbb{R}^1$$

When training with the PPO pipeline, the Gymnasium wrappers `RescaleAction` and `NormalizedObservation` from `src/rl_missile_sim/utils.py` are composed around the environment to normalize actions to $[-1, 1]$ and observation channels to $[0, 1]$.

---

## Guidance Policy Derivations

All guidance policies in this codebase output a scalar command $a_n$ representing acceleration normal to the missile's velocity vector $\mathbf{v}_M$. Let the normal unit vector be:

$$\hat{\mathbf{n}}_M = \begin{bmatrix} -\sin\psi_M \\ \cos\psi_M \end{bmatrix} = \frac{1}{\|\mathbf{v}_M\|} \begin{bmatrix} -E_{v, M} \\ N_{v, M} \end{bmatrix}$$

---

### 1. Proportional Navigation (ProNav / PN)

Proportional Navigation is the foundational guidance law in aerospace engineering. The objective of PN is to drive the inertial LOS angular rate to zero ($\dot{\theta} \to 0$).

#### Mathematical Derivation

When $\dot{\theta} = 0$, the Line-Of-Sight maintains a constant orientation in inertial space while range decreases—a condition known in maritime navigation as **Constant Bearing Decreasing Range (CBDR)**. Kinematically, this guarantees a collision course.

In **Pure Proportional Navigation (PPN)**, the commanded acceleration is directed normal to the missile velocity vector:

$$a_n = N' \, V_c \, \dot{\lambda}$$

where:
- $N'$ is the dimensionless effective navigation ratio ($3 \le N' \le 5$, chosen as $N' = 4$ in `src/rl_missile_sim/eval_train.py`)
- $V_c = \text{obs}[14]$ is the closing velocity ($-\dot{r}$)
- $\dot{\lambda} = \text{obs}[16]$ is the LOS rate ($\dot{\theta}$)

```python
def pronav_policy(obs):
    # obs[12]: V_M, obs[14]: closing velocity (V_c), obs[16]: LOS rate (lambda_dot)
    action = 4.0 * obs[12] * obs[16]
    return action
```

#### Collision Triangle and Stability

Consider the kinematic relation between the missile velocity heading error $\eta = \psi_M - \theta$ and the LOS rate $\dot{\theta}$. Differentiating $r \dot{\theta} + 2 \dot{r} \dot{\theta} = a_T - a_M^\perp$ along the engagement shows that for a non-accelerating target ($a_T = 0$), the linearized differential equation governing $\dot{\theta}$ is:

$$\frac{d\dot{\theta}}{dt} = -\frac{(N' - 2) V_c}{r} \dot{\theta}$$

Integrating this relation from $t=0$ to $t_f$ with $r(t) = V_c (t_f - t)$ yields:

$$\dot{\theta}(t) = \dot{\theta}(0) \left(1 - \frac{t}{t_f}\right)^{N' - 2}$$

- If $N' > 2$, $\dot{\theta}(t) \to 0$ as $t \to t_f$, guaranteeing zero terminal miss distance.
- If $N' > 3$, the commanded acceleration $a_n(t) \propto \dot{\theta}(t)$ asymptotically decays to zero at intercept, avoiding late-game actuator saturation.

---

### 2. Linear Quadratic Regulator (LQR) Guidance

LQR guidance frames interception as an infinite-horizon continuous-time optimal regulator problem over linearized relative kinematics.

#### State-Space Double-Integrator Formulation

Let the Cartesian relative position and relative velocity errors in North and East coordinates be:

$$\mathbf{x} = \begin{bmatrix} R_N \\ \Delta v_N \\ R_E \\ \Delta v_E \end{bmatrix} = \begin{bmatrix} N_T - N_M \\ N_{v, T} - N_{v, M} \\ E_T - E_M \\ E_{v, T} - E_{v, M} \end{bmatrix} \in \mathbb{R}^4$$

Assuming the target maintains constant velocity ($\dot{\mathbf{v}}_T = \mathbf{0}$) and the missile accelerates according to control vector $\mathbf{u} = [u_N, u_E]^T \in \mathbb{R}^2$, the relative equations of motion decouple into two orthogonal double integrators:

$$\dot{\mathbf{x}} = \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{u}$$

$$\mathbf{A} = \begin{bmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0
\end{bmatrix}, \quad
\mathbf{B} = \begin{bmatrix}
0 & 0 \\
-1 & 0 \\
0 & 0 \\
0 & -1
\end{bmatrix} \quad \left(\text{in relative coords: } \dot{\mathbf{v}}_{\text{rel}} = -\mathbf{a}_M\right)$$

In `eval_train.py`, the signs are parameterized such that $\mathbf{B} = \begin{bmatrix} 0 & 0 \\ 1 & 0 \\ 0 & 0 \\ 0 & 1 \end{bmatrix}$ with control $\mathbf{u}_{\text{acc}} = -\mathbf{K}\mathbf{x}$.

#### Cost Functional and Riccati Equation

We define the quadratic performance index balancing terminal interception error and control expenditure:

$$J = \int_0^\infty \left(\mathbf{x}^T \mathbf{Q} \mathbf{x} + \mathbf{u}^T \mathbf{R} \mathbf{u}\right) dt$$

where the state penalty $\mathbf{Q} \succeq 0$ and control penalty $\mathbf{R} \succ 0$ are:

$$\mathbf{Q} = \text{diag}(q_p, q_v, q_p, q_v) = \begin{bmatrix}
10 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 10 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \quad
\mathbf{R} = \text{diag}(1, 1)$$

The steady-state optimal feedback gain is obtained by solving the Continuous-Time Algebraic Riccati Equation (CARE) for positive-definite matrix $\mathbf{P}$:

$$\mathbf{A}^T \mathbf{P} + \mathbf{P} \mathbf{A} - \mathbf{P} \mathbf{B} \mathbf{R}^{-1} \mathbf{B}^T \mathbf{P} + \mathbf{Q} = \mathbf{0}$$

Using `scipy.linalg.solve_continuous_are`, the optimal linear state-feedback gain matrix is computed as:

$$\mathbf{K} = \mathbf{R}^{-1} \mathbf{B}^T \mathbf{P}$$

$$\mathbf{u}_{\text{vec}} = -\mathbf{K} \mathbf{x} = \begin{bmatrix} u_N \\ u_E \end{bmatrix}$$

#### Normal Acceleration Projection

Since the missile can only maneuver via normal acceleration (constant scalar speed $V_M$), the 2D Cartesian acceleration vector $\mathbf{u}_{\text{vec}}$ is projected onto the normal unit vector $\hat{\mathbf{n}}_M$:

$$\hat{\mathbf{n}}_M = \begin{bmatrix} -v_E / \|\mathbf{v}\| \\ v_N / \|\mathbf{v}\| \end{bmatrix}$$

$$a_n = \mathbf{u}_{\text{vec}} \cdot \hat{\mathbf{n}}_M = u_N \left(-\frac{E_{v, M}}{V_M}\right) + u_E \left(\frac{N_{v, M}}{V_M}\right)$$

---

### 3. Optimal Guidance Law (OGL / Zero Effort Miss)

Optimal Guidance Law is derived from finite-horizon optimal control theory, minimizing terminal miss distance while penalizing total control effort.

#### Concept of Zero-Effort-Miss (ZEM)

The Zero-Effort-Miss vector $\mathbf{ZEM}(t)$ is defined as the distance by which the missile will miss the target if no further control inputs are applied from time $t$ until the final time $t_f$ ($a_M(\tau) = 0$ for $\tau \in [t, t_f]$).

Under constant-velocity kinematics for both vehicles, the projected positions at intercept time $t_f = t + t_{\text{go}}$ are:

$$\mathbf{p}_M(t_f) = \mathbf{p}_M(t) + \mathbf{v}_M(t) t_{\text{go}}$$

$$\mathbf{p}_T(t_f) = \mathbf{p}_T(t) + \mathbf{v}_T(t) t_{\text{go}}$$

Subtracting the two equations gives the closed-form expression for ZEM:

$$\mathbf{ZEM}(t) = \mathbf{p}_T(t_f) - \mathbf{p}_M(t_f) = \mathbf{r}(t) + \mathbf{v}_{\text{rel}}(t) t_{\text{go}}$$

where the estimated time-to-go ($t_{\text{go}}$) is:

$$t_{\text{go}} = \frac{r(t)}{V_c(t)} = \frac{\|\mathbf{p}_T - \mathbf{p}_M\|^2}{-(\mathbf{p}_T - \mathbf{p}_M) \cdot (\mathbf{v}_T - \mathbf{v}_M)}$$

#### Optimal Control Derivation via Pontryagin's Principle

Consider the continuous-time optimal control problem:

$$\min_{\mathbf{a}_M} J = \frac{1}{2} s_f \|\mathbf{ZEM}(t_f)\|^2 + \frac{1}{2} \int_t^{t_f} \|\mathbf{a}_M(\tau)\|^2 d\tau$$

For a hard terminal intercept constraint ($s_f \to \infty \implies \mathbf{ZEM}(t_f) = \mathbf{0}$), Pontryagin's Minimum Principle yields:

$$\mathbf{a}_{\text{cmd}}(t) = \frac{N'}{t_{\text{go}}^2} \mathbf{ZEM}(t)$$

When derived to minimize terminal position and velocity errors simultaneously with fixed weight ratios, the generalized command vector implemented in `eval_train.py` is:

$$\mathbf{a}_{\text{opt}} = \frac{6}{t_{\text{go}}^2} (\mathbf{p}_T - \mathbf{p}_M) + \frac{4}{t_{\text{go}}} (\mathbf{v}_T - \mathbf{v}_M)$$

Notice that:
$$\frac{6}{t_{\text{go}}^2}\mathbf{r} + \frac{4}{t_{\text{go}}}\mathbf{v}_{\text{rel}} = \frac{6}{t_{\text{go}}^2}\left(\mathbf{r} + \mathbf{v}_{\text{rel}} t_{\text{go}}\right) - \frac{2}{t_{\text{go}}}\mathbf{v}_{\text{rel}} = \frac{6}{t_{\text{go}}^2}\mathbf{ZEM} - \frac{2}{t_{\text{go}}}\mathbf{v}_{\text{rel}}$$

The scalar normal command is obtained by projecting $\mathbf{a}_{\text{opt}}$ onto the vehicle's normal unit vector $\hat{\mathbf{n}}_M$:

$$a_n = \mathbf{a}_{\text{opt}} \cdot \hat{\mathbf{n}}_M$$

---

### 4. Deep Reinforcement Learning (PPO Policy)

The reinforcement learning guidance agent is formulated as a continuous-action Markov Decision Process (MDP):

$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

- **State Space $\mathcal{S} \subset \mathbb{R}^{33}$**: Fully observable kinematic vector + 16-channel LIDAR ranges.
- **Action Space $\mathcal{A} \subset [-1, 1]$**: Scaled normal acceleration command mapped linearly to $[-a_{\max}, a_{\max}]$ via `RescaleAction`.
- **Discount Factor $\gamma = 0.99$**.

#### Multi-Objective Reward Function

The reward function shapes intercept behavior, minimizes energy, penalizes target opening, and enforces obstacle avoidance:

$$R_t = R_{\text{intercept}} + R_{\text{range}} + R_{\text{effort}} + R_{\text{opening}} + R_{\text{obstacle}}$$

1. **Terminal Intercept Bonus ($R_{\text{intercept}}$)**:
   $$R_{\text{intercept}} = \begin{cases} +10.0, & \text{if } r \le r_{\text{collision}} \, (100\,\text{m}) \\ 0.0, & \text{otherwise} \end{cases}$$

2. **Range Proximity Penalty ($R_{\text{range}}$)**:
   Quadratic penalty encouraging the agent to close range continuously:
   $$R_{\text{range}} = -0.03 \left(\frac{r}{R_{\max}}\right)^2$$

3. **Control Effort Regularization ($R_{\text{effort}}$)**:
   Penalizes high commanded-g maneuvers to promote energy efficiency:
   $$R_{\text{effort}} = -0.02 \left(\frac{a_n}{a_{\max}}\right)^2$$

4. **Range Rate / Opening Penalty ($R_{\text{opening}}$)**:
   Penalizes trajectories where range increases ($\dot{r} > 0 \implies V_c < 0$):
   $$R_{\text{opening}} = \begin{cases} -0.01, & \text{if } \dot{r} > 0 \\ 0.0, & \text{otherwise} \end{cases}$$

5. **Obstacle Collision Penalty ($R_{\text{obstacle}}$)**:
   $$R_{\text{obstacle}} = \begin{cases} -10.0 \quad (\text{and episode terminates}), & \text{if } \|\mathbf{p}_M - \mathbf{p}_{o, j}\| < R_{o, j} + r_{\text{rad}, M} \\ 0.0, & \text{otherwise} \end{cases}$$

#### Proximal Policy Optimization (PPO) Formulation

The policy $\pi_\theta(a|s)$ and state-value estimator $V_\phi(s)$ are trained using PPO with Generalized Advantage Estimation (GAE). The clipped surrogate objective is:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right) \right]$$

where:
- Probability ratio: $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$
- Clipping hyperparameter: $\epsilon = 0.2$
- Advantage function: $\hat{A}_t = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$, with TD error $\delta_t^V = R_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$ and $\lambda = 0.95$

#### Neural Architecture Hyperparameters

- **Policy Architecture**: Multi-Layer Perceptron (MLP) with two hidden layers $[64, 64]$ and $\tanh$ activations.
- **Optimizer**: Adam ($\alpha = 5 \times 10^{-5}$).
- **Rollout Steps**: $n_{\text{steps}} = 2048$ across $n_{\text{envs}} = 20$ vectorized workers (batch size $512$).
- **Entropy Coefficient**: $c_{\text{ent}} = 0.01$ (ensuring thorough exploration of the angular acceleration envelope).

---

## Environment Dynamics, Obstacles, and Rewards

### Engagement Limits and Truncation

An episode terminates or truncates under the following mutually exclusive conditions:

- **Terminated (`terminated = True`)**:
  - **Hit / Interception**: $r \le r_{\text{collision}} = 100\,\text{m}$.
  - **Obstacle Impact**: The pursuer penetrates the perimeter of any obstacle cylinder:
    $$\|\mathbf{p}_M - \mathbf{p}_{o, j}\| \le R_{o, j} + r_{\text{rad}, M}$$
- **Truncated (`truncated = True`)**:
  - **Time-Out**: $k \ge k_{\max} = \frac{t_{\max}}{\Delta t} = \frac{15.0}{0.1} = 150\,\text{iterations}$.
  - **Out-of-Bounds**: $r \ge R_{\max} = 20{,}000\,\text{m}$.

### Obstacle Field Layout

The environment features four static circular exclusion zones:

$$\begin{aligned}
\mathcal{O}_1 &: [N=2{,}000\,\text{m}, \, E=2{,}000\,\text{m}], \quad R_1 = 2{,}000\,\text{m} \\
\mathcal{O}_2 &: [N=1{,}000\,\text{m}, \, E=-5{,}000\,\text{m}], \quad R_2 = 1{,}000\,\text{m} \\
\mathcal{O}_3 &: [N=-5{,}000\,\text{m}, \, E=7{,}000\,\text{m}], \quad R_3 = 2{,}000\,\text{m} \\
\mathcal{O}_4 &: [N=5{,}000\,\text{m}, \, E=-7{,}000\,\text{m}], \quad R_4 = 2{,}000\,\text{m}
\end{aligned}$$

---

## Installation and Quickstart

### Prerequisites

- Python $\ge 3.11$
- [uv](https://docs.astral.sh/uv/) (recommended high-performance package manager) or standard `pip`

### 1. Installation via uv

```bash
# Clone the repository
git clone https://github.com/jepierre/rl_missile_sim.git
cd rl_missile_sim

# Install core dependencies with RL training support (Stable-Baselines3, PyTorch, Gymnasium)
uv sync --extra train

# (Optional) Install pygame for hardware-accelerated 2D visualization
uv sync --extra render
```

### 2. CUDA GPU Acceleration on Windows

To install PyTorch with native CUDA acceleration, add the official PyTorch wheel index to your `pyproject.toml` (e.g. `cu129` or `cu130`):

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu129" }

[[tool.uv.index]]
name = "pytorch-cu129"
url = "https://download.pytorch.org/whl/cu129"
explicit = true
```

Then run:

```bash
uv lock
uv sync --extra train
```

Verify CUDA initialization:

```bash
uv run python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
```

---

## CLI Reference

### Policy Evaluation

Evaluate any guidance policy in the simulation without opening a GUI window:

```bash
# Proportional Navigation (Pure PN)
uv run rl-missile-sim-eval --mode eval --policy pronav --no-render --episodes 10

# Optimal Guidance Law (ZEM Guidance)
uv run rl-missile-sim-eval --mode eval --policy optimal --no-render --episodes 10

# Linear Quadratic Regulator
uv run rl-missile-sim-eval --mode eval --policy lqr --no-render --episodes 10

# Random Action Baseline
uv run rl-missile-sim-eval --mode eval --policy random --no-render --episodes 5

# Trained Reinforcement Learning Neural Policy
uv run rl-missile-sim-eval --mode eval --policy agent --agent_path logs/ppo_missile/final_model.zip --no-render
```

To run with live Pygame visual feedback, omit the `--no-render` flag.

### Training PPO Agents

Launch vectorized PPO training across $20$ parallel environment instances:

```bash
uv run rl-missile-sim-eval --mode train --timesteps 1100000 --n_envs 20
```

Checkpoints and TensorBoard event logs are automatically recorded under `logs/ppo_missile_<timestamp>_<git_hash>/`.

### Interactive Pygame Demonstration

Launch the tactical radar display demo driven by ProNav guidance:

```bash
uv run rl-missile-sim-render
```

### Running Unit Tests

Execute the automated test suite covering Gymnasium compliance, observation invariants, and reward bounds:

```bash
# Run all unit tests
uv run python -m unittest discover -s tests -v

# Run single targeted test
uv run python -m unittest tests.unit.test_missile_sim.TestMissileSimEnvCheck.test_observation_space -v
```

---

## Project Architecture

```
rl_missile_sim/
├── src/
│   └── rl_missile_sim/
│       ├── __init__.py           # Package entry point & Gymnasium ID registration ('missile_sim-v0')
│       ├── eval_train.py         # SB3 PPO training pipeline & classical guidance implementations
│       ├── render_demo.py        # Pygame standalone interactive tactical display
│       ├── utils.py              # NormalizedObservation & RescaleAction wrappers
│       └── envs/
│           ├── __init__.py       # Environment module exports
│           └── missile_sim.py    # Core MissileSim Gymnasium environment & Entity kinematics
├── tests/
│   └── unit/
│       ├── __init__.py
│       └── test_missile_sim.py   # SB3 check_env, observation & reward invariants
├── pyproject.toml                # PEP 621 metadata, dependencies & console scripts
├── uv.lock                       # Deterministic dependency lockfile
├── AGENTS.md                     # Agent conventions and technical execution guide
└── README.md                     # Technical documentation & GNC derivations
```

---

## References

1. **Zarchan, P.** (2012). *Tactical and Strategic Missile Guidance* (6th ed.). Progress in Astronautics and Aeronautics, AIAA.
2. **Palumbo, N. F., Blauwkamp, R. A., & Cannizzaro, J. M.** (2010). [Introduction to Missile Guidance](https://www.jhuapl.edu/Content/techdigest/pdf/V29-N01/29-01-Palumbo_GuestEditor.pdf). *Johns Hopkins APL Technical Digest*, 29(1), 3–18.
3. **Palumbo, N. F., Blauwkamp, R. A., & Cannizzaro, J. M.** (2010). [Basic Principles of Homing Guidance](https://secwww.jhuapl.edu/techdigest/content/techdigest/pdf/V29-N01/29-01-Palumbo_Homing.pdf). *Johns Hopkins APL Technical Digest*, 29(1), 25–41.
4. **Stevens, B. L., Lewis, F. L., & Johnson, E. N.** (2015). *Aircraft Control and Simulation: Dynamics, Controls Design, and Autonomous Systems* (3rd ed.). John Wiley & Sons.
5. **Gaudet, B., Linares, R., & Furfaro, R.** (2020). [Deep Reinforcement Learning for Exo-Atmospheric Interception with Uncertain Dynamics](https://arc.aiaa.org/doi/10.2514/1.I010970). *Journal of Guidance, Control, and Dynamics*, 43(8).
6. **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). *arXiv:1707.06347*.
