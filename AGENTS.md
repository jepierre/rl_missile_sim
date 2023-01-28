# Agent instructions

## Project layout

- The installable Python package uses the `src/` layout: application code is under `src/rl_missile_sim/`.
- `src/rl_missile_sim/envs/missile_sim.py` implements the simulation and Gymnasium environment.
- `src/rl_missile_sim/eval_train.py` provides the Stable-Baselines3 PPO training and evaluation CLI, including the baseline guidance policies.
- `src/rl_missile_sim/utils.py` contains shared action and observation wrappers. `render_demo.py` provides the optional pygame viewer.
- Unit tests live under `tests/unit/`.
- Dependencies and console scripts are declared in `pyproject.toml`; `uv.lock` is the uv lockfile.

## Build and test commands

- Install core and training/test dependencies: `uv sync --extra train`.
- Build source and wheel distributions: `uv build`.
- Run the unit test suite: `uv run python -m unittest discover -s tests -v`.
- Run a specific test: `uv run python -m unittest tests.unit.test_missile_sim.TestMissileSimEnvCheck.test_observation_space -v`.
- Install the optional pygame dependency with `uv sync --extra render`.
- No lint command is configured.

## Application conventions

- Import the installed package by its name, `rl_missile_sim`; use relative imports between modules inside that package.
- The environment follows the Gymnasium API: `reset()` returns `(observation, info)` and `step()` returns `(observation, reward, terminated, truncated, info)`. Keep observation/action values consistent with their declared spaces.
- State and coordinates are planar north/east positions in meters; headings are radians and simulation time is seconds. The missile action is a scalar normal-acceleration command, bounded by `amax`, that changes heading while entity speed remains constant.
- The observation is a flat vector: pursuer state `[N, E, psi, Nv, Ev, radius]`, target state in the same order, followed by `[pursuer speed, range, closing velocity, line-of-sight angle, line-of-sight rate]`; lidar readings are appended when enabled. Built-in guidance policies rely on these indices.
- `RescaleAction` and `NormalizedObservation` in `utils.py` are composed around the environment in the PPO path; wrapper transforms and declared spaces must stay consistent.
- The package registers `missile_sim-v0` with `gymnasium.register`, and the environment class is a `gymnasium.Env`. Keep registration, seeding (`super().reset(seed=...)` + `self.np_random`), and rendering (`render_mode` at construction, no `mode` argument) on the modern Gymnasium API.
