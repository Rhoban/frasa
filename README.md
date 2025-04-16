
# FRASA: Fall Recovery And Stand up Agent

This repository contains the code to train the **Fall Recovery And Stand-up Agent (FRASA)** using reinforcement learning (RL) algorithms. Training is conducted in a *MuJoCo* environment with the *stable-baselines3* library. The agent is designed to recover from falls and stand up from both prone and supine positions.

The methodology and results of this work are detailed in [this paper](https://arxiv.org/pdf/2410.08655), and a comprehensive video presentation is available [here](https://youtu.be/VRgBEy0yhfU).

The platform used for this project is the *Sigmaban humanoid robot*, developed by the *Rhoban Team*. The performance of the trained agent has been validated on the physical robot, as demonstrated in the examples below:

<div align=center />
   <img src="https://github.com/user-attachments/assets/52333ee2-8c39-40f3-acb4-0d816a807dfa" alt="Pull_back" width="47%" maxwidth="500px">
   <img src="https://github.com/user-attachments/assets/6e015b8b-a0ed-45c1-be39-5183c4b9a0f5" alt="Pull_front" width="47%" maxwidth="500px">
</div>

## Installation

First of all, you need to install RL Baselines3 Zoo to train and enjoy agents:

```
pip install rl_zoo3
```

Then, you can install frasa-env from source using the following command:

```
pip install -e .
```

### Issue with MuJoCo using Wayland

If you are using Wayland (instead of X11), you may encounter issues with the MuJoCo viewer,
such as frozen windows or buttons not working. To solve this, you can build
GLFW from source with shared libraries and set the `LD_LIBRARY_PATH` and
`PYGLFW_LIBRARY` environment variables to point to the built libraries.

1. Download source package here and unzip it: https://www.glfw.org/download.html

2. Install dependancies to build GLFW for Wayland and X11

```
sudo apt install libwayland-dev libxkbcommon-dev xorg-dev
```

3. Go to the glfw extracted folder and use the following cmake command to build it with shared libs

```
cd path/to/glfw
cmake -S . -B build -D BUILD_SHARED_LIBS=ON
```

4. Build GLFW from source

```
cd path/to/glfw/build
make
```

5. Change LD_LIBRARY_PATH and PYGLFW_LIBRARY to match GLFW version you built and add it to your bashrc

```
export LD_LIBRARY_PATH="LD_LIBRARY_PATH:path/to/glfw/build/src/"
export PYGLFW_LIBRARY="path/to/glfw/build/src/libglfw.so"
```

## Environment Setup

### Generating initial positions

Pre-generating initial positions for the standup environment is recommended,
as it can be time-consuming to generate them during training. To do so, you can
use the standup_generate_initial.py script:

```bash
python standup_generate_initial.py
```

It will generate initial positions by letting the robot fall from random positions
and store them in `frasa_env/env/standup_initial_configurations.pkl`.
Let the script run until you have collected enough initial positions
(typically a few thousand). You can stop the script at any time using Ctrl+C;
the generated positions will be saved automatically.

<img src="https://github.com/user-attachments/assets/876ea026-487d-4c27-8c95-98c2bbacc8ee" width="40%" align="right">

### Testing the environment

You can use the `test_env.py` script to test the environment:

```bash
# Start the environment without action
python test_env.py

# Start the environment with noise on the actions
python test_env.py --normal --std 2
```

## Training

You can train an agent using:

```bash
python train_sbx.py \
    --algo crossq \
    --env frasa-standup-v0 \
    --conf hyperparams/crossq.yml
```

Where the arguments are:
* `algo`: The RL algorithm to use
* `env`: The environment to train on
* `conf`: The hyperparameters file to use

The hyperparameters for the environment are defined in `hyperparams/[algo].yml`.

The trained agent will be stored in `logs\[algo]\[env]_[exp-id]`.

## Enjoy a Trained Agent

If a trained agent exists, you can see it in action using:

```bash
python enjoy_sbx.py \
    --algo crossq \
    --env frasa-standup-v0 \
    --gym-packages frasa_env \
    --folder logs/ \
    --load-best \
    --exp-id 0
```

<img src="https://github.com/user-attachments/assets/05655c5e-64e9-48f4-8f53-d401a03ba40e" align="right" height="250px">

Where the arguments are:

* `algo`: The RL algorithm to use
* `env`: The environment to enjoy on
* `gym-packages`: The gym package used to register the environment
* `folder`: The folder where the trained agent is stored
* `load-best`: Used to load the best agent.
* `exp-id 0`: The experiment ID to use (`0` meaning the latest).

## Sigmaban Model

<img src="https://github.com/user-attachments/assets/0fab475c-8b60-48e9-aebe-0ca0506e3917" align=left height="260px">

<br>

<br>

The Sigmaban robot is a small humanoid developed by the Rhoban team to
compete in the RoboCup KidSize league. The robot is 70 cm tall and weighs
7.5 kg. It has 20 degrees of freedom, a camera, and is equipped with pressure
sensors and an IMU.

The MuJoCo XML model of this robot is located in the `frasa_env/mujoco_simulator/model` folder.

For a detailed description of the process to convert URDF to MuJoCo XML,
see the [README](frasa_env/mujoco_simulator/model/README.md) in the model directory.

<br>

<br>

## Stand Up Environment

The stand up environment can be found in `frasa_env/env/standup_env.py`.

<img src="https://github.com/user-attachments/assets/053fce01-3255-4ad1-9377-e576d38ce44c" align="right" width="50%">

The environment simplifies the learning process by controlling only the 5 DoFs presented on the right (elbow, shoulder_pitch, hip_pitch, knee, ankle_pitch). These are the primary joints involved in recovery and standing movements in the sagittal plane $(x, z)$. The actions are symmetrically applied to both sides of the robot.

The robot's state is characterized by the **pose vector** $\psi$, which includes the joint angles $q$ and the trunk pitch $\theta$. The target pose for recovery is defined as:

$$ \psi_{\text{target}} = [q_{\text{target}}, \theta_{\text{target}}] $$

This target pose represents the upright position the robot should achieve after recovery.

At the start of each episode, the robot’s joint angles $q$ and trunk pitch $\theta$ are set to random values within their
physical limits. The robot is then released above the ground, simulating various fallen states. The physical simulation is
then stepped using current joints configuration as targets, until stability is reached, at which point
the learning process begins.

An episode termination occurs when the robot reaches an unsafe or irreversible state:
- Trunk pitch ($\theta$) exceeds a predefined threshold, indicating the robot is upside down.
- Trunk pitch velocity ($\dot{\theta}$) surpasses a critical threshold, suggesting very violent displacements of the pelvis.

An episode is truncated if its duration exceed a predefined maximum duration. This allows for an increase in the diversity of encountered states.

The robot can be controlled using 3 modes: `position`, `velocity` and `error`. The recommended and used control mode for the experiments in the paper is `velocity`. This mode interprets the actions as the rate of change for the desired joint angles, ensuring smooth and adaptive movements.

### Observation Space

The observation space for the **StandupEnv** captures key elements of the robot's posture, movements, and control state. It includes:

| **Index** | **Observation** | **Range** | **Description** |
|---|---|---|---|
| 0-4 | `q` | [-π, π] | Joint angles for the 5 degrees of freedom (`elbow`, `shoulder_pitch`, `hip_pitch`, `knee`, `ankle_pitch`) |
| 5-9 | `dq` | [-10, 10] | Joint angular velocities for the 5 DoFs |
| 10-14 | `ctrl` | Actuator control range | Actuator control values for the joints |
| 15 | `tilt` | [-π, π] | Trunk pitch angle |
| 16 | `dtilt` | [-10, 10] | Trunk pitch angular velocity |
| 17+ | `previous_actions` | Action space range | History of the most recent actions, defined by `options["previous_actions"]` |

### Action Space

The action space specifies control commands for the robot's 5 joints. Its structure depends on the control mode (`position`, `velocity`, or `error`) specified in `options["control"]`.

| **Mode** | **Description** | **Range** | **Notes** |
|---|---|---|---|
| `position` | Absolute joint positions | [range_low, range_high] | Actuator range defined in the `model/robot.xml` |
| `velocity` | Rate of change for joint positions | [-vmax, vmax] | Maximum velocity vmax is 2π rad/s |
| `error` | Desired deviation from current joint positions | [-π/4, π/4] | Represents a proportional error to adjust the current state |

In all modes, the action space corresponds to the robot's 5 controlled joints.

### Reward

The reward function incentivizes the robot to achieve and maintain the desired upright posture while avoiding abrupt movements and collisions. It is defined as:

$$ R = R_{state} + R_{variation} + R_{collision} $$

1. State Proximity Reward

The state proximity reward $R_{state}$ represents the proximity to the desired state $\psi_{target}$:

$$ R_{state} = \text{exp}\left(-20 \cdot \left| \psi_{current} - \psi_\{target} \right|^2 \right) $$

2. Variation Penalty Reward

The variation penalty reward $R_{variation}$ penalizes large differences between consecutive actions:

$$ R_{variation} = \text{exp}\left( -\left| a_t - a_{t-1} \right| \right) \cdot 0.05 $$

Where $a_t$ and $a_{t-1}$ corresponds respectivelly to the current and previous action vectors, which depend of the control mode.

3. No Self-collision Reward

The no self-collision reward $R_{collision}$ penalizes self-collisions detected by the simulation:

$$ R_{collision} = \text{exp}\left(-\text{collisions()} \right) \cdot 0.1 $$

Where **collision()** corresponds to `sim.self_collisions()` function.

Note that additional penalties and smoothness incentives are applied depending on the control mode. For a detailed implementation, please refer to the source code in `frasa_env/env/standup_env.py`.

### Environment Options

The **StandupEnv** environment includes a variety of configurable options to tailor the training and evaluation process. Below is a description of the key parameters:

| **Option** | **Description** | **Default Value** |
|---|---|---|
| `stabilization_time` | Duration of the stabilization pre-simulation to let the robot stabilize under gravity (in seconds) | `2.0` |
| `truncate_duration` | Maximum duration of an episode before truncation (in seconds) | `5.0` |
| `dt` | Time interval for applying agent control commands (in seconds) | `0.05` |
| `vmax` | Maximum command angular velocity for the robot's joints (in rad/s) | `2 * π` |
| `render_realtime` | If `True`, rendering occurs in real-time during simulation | `True` |
| `desired_state` | Target state for the robot, defined as joint angles and IMU pitch (in radians) | `[-49.5°, 19.5°, -52°, 79°, -36.5°, -8.5°]` |
| `reset_final_p` | Probability of resetting the robot in its final (stable) position | `0.1` |
| `terminate_upside_down` | Whether to terminate the episode if the robot is upside down | `True` |
| `terminate_gyro` | Whether to terminate the episode based on abnormal gyroscopic readings | `True` |
| `terminate_shock` | Whether to terminate the episode due to excessive shocks or impacts | `False` |
| `random_angles` | Randomization range for initial joint angles (in degrees) | `±1.5` |
| `random_time_ratio` | Randomization range for time scaling during simulation | `[0.98, 1.02]` |
| `random_friction` | Randomization range for surface friction | `[0.25, 1.5]` |
| `random_body_mass` | Randomization range for body part masses (relative ratio) | `[0.8, 1.2]` |
| `random_body_com_pos` | Randomization range for the center of mass positions (in meters) | `±5e-3` |
| `random_damping` | Randomization range for joint damping (relative ratio) | `[0.8, 1.2]` |
| `random_frictionloss` | Randomization range for friction loss (relative ratio) | `[0.8, 1.2]` |
| `random_v` | Randomization range for voltage supply to the motors (in volts) | `[13.8, 16.8]` |
| `nominal_v` | Nominal voltage supply to the motors (in volts) | `15` |
| `control` | Type of control applied to the robot: `position`, `velocity`, or `error` | `velocity` |
| `interpolate` | Whether to interpolate control commands for smoother execution | `True` |
| `qdot_delay` | Delay applied to joint velocity feedback (in seconds) | `0.030` |
| `tilt_delay` | Delay applied to tilt angle feedback (in seconds) | `0.050` |
| `dtilt_delay` | Delay applied to tilt velocity feedback (in seconds) | `0.050` |
| `previous_actions` | Number of previous actions considered for state representation | `1` |

These options can be customized by passing a dictionary of parameters to the environment during initialization. Example:

```python
env = StandupEnv(options={
    "truncate_duration": 10.0,
    "random_angles": 2.0,
    "terminate_shock": True
})
```

## Citing

To cite this repository in publications:

```bibtex
@article{frasa2024,
   title={FRASA: An End-to-End Reinforcement Learning Agent for Fall Recovery and Stand Up of Humanoid Robots},
   author={Clément Gaspard and Marc Duclusaud and Grégoire Passault and Mélodie Daniel and Olivier Ly},
   year={2024},
   eprint={2410.08655},
   archivePrefix={arXiv},
   primaryClass={cs.RO},
   url={https://arxiv.org/abs/2410.08655},
}
```
