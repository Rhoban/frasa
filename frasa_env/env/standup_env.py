import os
import pickle
import random
import warnings
from typing import Optional

import gymnasium
import numpy as np
from gymnasium import spaces

from frasa_env.mujoco_simulator.simulator import Simulator, tf


class StandupEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 30}

    def __init__(self, render_mode="none", options: Optional[dict] = None, evaluation: bool = False):
        self.options = {
            # Duration of the stabilization pre-simulation (waiting for the gravity to stabilize the robot) [s]
            "stabilization_time": 2.0,
            # Duration of the episode before truncation [s]
            "truncate_duration": 5.0,
            # Period for the agent to apply control [s]
            "dt": 0.05,
            # Maximum command angular velocity [rad/s]
            "vmax": 2 * np.pi,
            # Is the render done in "realtime"
            "render_realtime": True,
            # Target robot state (q_motors, tilt) [rad^6]
            # [elbow, shoulder_pitch, hip_pitch, knee, ankle_pitch, IMU_pitch]
            "desired_state": np.deg2rad([-49.5, 19.5, -52, 79, -36.5, 8.5]),
            # Probability of seeding the robot in finale position
            "reset_final_p": 0.1,
            # Termination conditions
            "terminate_upside_down": True,
            "terminate_gyro": True,
            "terminate_shock": False,
            # Randomization
            "random_angles": 1.5,  # [+- deg]
            "random_time_ratio": [0.98, 1.02],  # [+- ratio]
            "random_friction": [0.25, 1.5],  # [friction]
            "random_body_mass": [0.8, 1.2],  # [+- ratio]
            "random_body_com_pos": 5e-3,  # [+- m]
            "random_damping": [0.8, 1.2],  # [+- ratio]
            "random_frictionloss": [0.8, 1.2],  # [+- ratio]
            "random_v": [13.8, 16.8],  # [volts]
            "nominal_v": 15,  # [volts]
            # Control type (position, velocity or error)
            "control": "velocity",
            "interpolate": True,
            # Delay for velocity [s]
            "qdot_delay": 0.030,
            "tilt_delay": 0.050,
            "dtilt_delay": 0.050,
            # Previous actions
            "previous_actions": 1,
        }
        self.options.update(options or {})

        self.render_mode = render_mode
        self.sim = Simulator()

        # Degrees of freedom involved
        self.dofs = ["elbow", "shoulder_pitch", "hip_pitch", "knee", "ankle_pitch"]
        self.ranges = [self.sim.model.actuator(f"left_{dof}").ctrlrange for dof in self.dofs]

        # Pre-fetching indexes and sites for faster evaluation
        self.trunk_site = self.sim.data.site("trunk")
        self.left_actuators_indexes = [self.sim.get_actuator_index(f"left_{dof}") for dof in self.dofs]
        self.right_actuators_indexes = [self.sim.get_actuator_index(f"right_{dof}") for dof in self.dofs]
        self.range_low = np.array([r[0] for r in self.ranges])
        self.range_high = np.array([r[1] for r in self.ranges])

        # Action space is q
        max_variation = self.options["dt"] * self.options["vmax"]
        self.delta_max = np.array([max_variation] * len(self.dofs))

        if self.options["control"] == "position":
            self.action_space = spaces.Box(
                np.array(self.range_low),
                np.array(self.range_high),
                dtype=np.float32,
            )
        elif self.options["control"] == "velocity":
            self.action_space = spaces.Box(
                np.array([-self.options["vmax"]] * len(self.dofs), dtype=np.float32),
                np.array([self.options["vmax"]] * len(self.dofs), dtype=np.float32),
                dtype=np.float32,
            )
        elif self.options["control"] == "error":
            self.action_space = spaces.Box(
                np.array([-np.pi / 4] * len(self.dofs), dtype=np.float32),
                np.array([np.pi / 4] * len(self.dofs), dtype=np.float32),
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unknown control type: {self.options['control']}")

        # Observation is q_pitch, dqpitch, tilt, dtilt
        self.observation_space = spaces.Box(
            np.array(
                [
                    # q (0-4)
                    *(-np.pi * np.ones(len(self.dofs))),
                    # dq (5-9)
                    *(-10 * np.ones(len(self.dofs))),
                    # ctrl (10-14)
                    *self.range_low,
                    # tilt (15)
                    -np.pi,
                    # dtilt (16)
                    -10,
                    # Previous action
                    *(list(self.action_space.low) * self.options["previous_actions"]),
                ],
                dtype=np.float32,
            ),
            np.array(
                [
                    # q (0-4)
                    *(np.pi * np.ones(len(self.dofs))),
                    # dq (5-9)
                    *(10 * np.ones(len(self.dofs))),
                    # ctrl (10-14)
                    *self.range_high,
                    # tilt (15)
                    np.pi,
                    # dtilt (16)
                    10,
                    # Previous action
                    *(list(self.action_space.high) * self.options["previous_actions"]),
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Window for qdot delay simulation
        self.q_history = []
        self.q_history_size = max(1, round(self.options["qdot_delay"] / self.sim.dt))

        # Window for tilt delay simulation
        self.tilt_history = []
        self.tilt_history_size = max(1, round(self.options["tilt_delay"] / self.sim.dt))

        self.dtilt_history = []
        self.dtilt_history_size = max(1, round(self.options["dtilt_delay"] / self.sim.dt))

        # Values for randomization
        self.trunk_body = "torso_2023"
        self.trunk_mass = self.sim.model.body(self.trunk_body).mass.copy()
        self.trunk_ipos = self.sim.model.body(self.trunk_body).ipos.copy()
        self.gainprm_original = self.sim.model.actuator_gainprm.copy()
        self.biasprm_original = self.sim.model.actuator_biasprm.copy()
        self.damping_original = self.sim.model.dof_damping.copy()
        self.frictionloss_original = self.sim.model.dof_frictionloss.copy()
        self.forcerange_original = self.sim.model.actuator_forcerange.copy()
        self.body_quat_original = self.sim.model.body_quat.copy()

        # Loading initial configuration cache
        initial_config_path = self.get_initial_config_filename()
        self.initial_config = None
        if os.path.exists(initial_config_path):
            # print(f"Loading initial configurations from {initial_config_path}")
            with open(initial_config_path, "rb") as f:
                self.initial_config = pickle.load(f)

    def get_initial_config_filename(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "standup_initial_configurations.pkl")

    def apply_control(self, q: np.ndarray, reset: bool = False) -> None:
        self.sim.data.ctrl[self.left_actuators_indexes] = q
        self.sim.data.ctrl[self.right_actuators_indexes] = q

        if reset:
            for k, dof in enumerate(self.dofs):
                self.sim.set_control(f"left_{dof}", q[k], True)
                self.sim.set_control(f"right_{dof}", q[k], True)

    def get_tilt(self) -> float:
        R = self.trunk_site.xmat
        return -np.arctan2(R[6], R[8])

    def get_observation(self) -> np.ndarray:
        # Retrieving joints
        q = self.q_history[-1]
        q_dot = (np.array(self.q_history[-1]) - np.array(self.q_history[0])) / (self.q_history_size * self.sim.dt)

        # Current control
        ctrl = [self.sim.get_control(f"left_{dof}") for dof in self.dofs]
        ctrl = np.array(ctrl)

        # Tilt, dtilt
        tilt = self.tilt_history[0]
        dtilt = self.dtilt_history[0]
        # gyro = self.sim.get_gyro()
        # dtilt = gyro[1]

        return np.array(
            [
                *q,
                *q_dot,
                *ctrl,
                tilt,
                dtilt,
                *(np.array(self.previous_actions).flatten()),
            ],
            dtype=np.float32,
        )

    def step(self, action):
        action = np.array(action)

        # Current control
        start_ctrl = [self.sim.get_control(f"left_{dof}") for dof in self.dofs]

        # Applying the action
        if self.options["control"] == "position":
            # Action is a q
            target_ctrl_unclipped = action
        elif self.options["control"] == "velocity":
            # Action is a q variation
            target_ctrl_unclipped = start_ctrl + action * self.options["dt"]
        elif self.options["control"] == "error":
            # Action is an error w.r.t the read configuration
            start_q = [self.sim.get_q(f"left_{dof}") for dof in self.dofs]
            target_ctrl_unclipped = np.array(start_q) + action

        # Limiting the control
        target_ctrl = np.clip(
            target_ctrl_unclipped,
            start_ctrl - self.delta_max * self.time_ratio,
            start_ctrl + self.delta_max * self.time_ratio,
        )

        # Limiting the control to the range
        target_ctrl = np.clip(target_ctrl, self.range_low, self.range_high)

        # Not terminating episode
        done = False
        shock = False

        # Step the simulation
        timesteps = round(self.time_ratio * self.options["dt"] / self.sim.dt)
        for k in range(timesteps):
            if self.options["interpolate"]:
                alpha = (k + 1) / timesteps
                self.apply_control(start_ctrl + alpha * (target_ctrl - start_ctrl))
            else:
                self.apply_control(target_ctrl)
            self.sim.step()

            if self.options["terminate_shock"]:
                centroidal_force = self.sim.centroidal_force()
                if centroidal_force - self.centroidal_force > 200.0:
                    shock = True
                self.centroidal_force = centroidal_force

            q = [self.sim.get_q(f"left_{dof}") for dof in self.dofs]
            self.q_history.append(q)

            self.tilt_history.append(self.get_tilt())
            self.dtilt_history.append(self.sim.get_gyro()[1])

            if self.render_mode == "human":
                self.sim.render(self.options["render_realtime"])

        # Terminating in case of upside down robot
        if self.options["terminate_upside_down"]:
            tilt = self.get_tilt()
            if np.rad2deg(np.abs(tilt)) > 135:
                done = True

        # Penalizing high gyro
        if self.options["terminate_gyro"]:
            gyro = self.sim.get_gyro()
            if abs(gyro[1]) > 5:
                done = True

        # Shock termination
        if self.options["terminate_shock"] and shock:
            done = True

        self.q_history = self.q_history[-self.q_history_size :]
        self.tilt_history = self.tilt_history[-self.tilt_history_size :]
        self.dtilt_history = self.dtilt_history[-self.dtilt_history_size :]

        # Extracting observation
        obs = self.get_observation()

        state_current = [*self.q_history[-1], self.tilt_history[-1]]
        reward = np.exp(-20 * (np.linalg.norm(np.array(state_current) - np.array(self.options["desired_state"])) ** 2))

        action_variation = np.abs(action - self.previous_actions[-1])
        self.previous_actions.append(action)
        self.previous_actions = self.previous_actions[-self.options["previous_actions"] :]

        # Penalizing velocities
        if self.options["control"] == "position":
            # Penalizing control velocity
            ctrl_velocity = (target_ctrl - start_ctrl) / self.options["dt"]
            reward += np.exp(-np.linalg.norm(ctrl_velocity)) * 1e-1

            # Penalizing action variation
            reward += np.exp(-np.linalg.norm(action_variation)) * 5e-2
        elif self.options["control"] == "velocity":
            # Penalizing action
            reward += np.exp(-np.linalg.norm(action)) * 1e-1

            # Penalizing action variation
            reward += np.exp(-np.linalg.norm(action_variation)) * 5e-2
        elif self.options["control"] == "error":
            # Penalizing action
            reward += np.exp(-np.linalg.norm(action)) * 1e-1

            # Penalizing action variation
            reward += np.exp(-np.linalg.norm(action_variation)) * 5e-2

        # No self collisions reward
        reward += np.exp(-self.sim.self_collisions()) * 1e-1

        # Terminating the episode after the trucate_duration
        terminated = self.sim.t > self.options["truncate_duration"]

        return obs, reward, done, terminated, {}

    def apply_angular_offset(self, joint: str, offset: float):
        body_id = self.sim.model.joint(joint).bodyid
        quat = self.sim.model.body_quat[body_id][0]
        quat = tf.quaternion_multiply(quat, tf.quaternion_about_axis(offset, [0, 0, 1]))
        self.sim.model.body_quat[body_id] = quat

    def apply_randomization(self):
        # Randomizing the time ratio
        self.time_ratio = self.np_random.uniform(*self.options["random_time_ratio"])

        # Randomizing the floor friction
        self.sim.set_floor_friction(self.np_random.uniform(*self.options["random_friction"]))

        # Randomizing the mass
        self.sim.model.body(self.trunk_body).mass = self.trunk_mass * self.np_random.uniform(*self.options["random_body_mass"])

        # Randomizing the center of mass
        self.sim.model.body(self.trunk_body).ipos = self.trunk_ipos + self.np_random.uniform(
            -self.options["random_body_com_pos"],
            self.options["random_body_com_pos"],
            size=3,
        )

        # Randomizing the voltage
        volts = self.np_random.uniform(*self.options["random_v"])
        volts_ratio = volts / self.options["nominal_v"]
        self.sim.model.actuator_gainprm = self.gainprm_original * volts_ratio
        self.sim.model.actuator_biasprm = self.biasprm_original * volts_ratio
        self.sim.model.actuator_forcerange = self.forcerange_original * volts_ratio

        # Randomize the damping
        damping_ratio = self.np_random.uniform(*self.options["random_damping"])
        self.sim.model.dof_damping = self.damping_original * damping_ratio

        frictionloss_ratio = self.np_random.uniform(*self.options["random_frictionloss"])
        self.sim.model.dof_frictionloss = self.frictionloss_original * frictionloss_ratio

        # Restoring body quats
        self.sim.model.body_quat = self.body_quat_original.copy()

        # Adding error angles
        err_rad = np.deg2rad(self.options["random_angles"])
        for dof in self.sim.dof_names():
            self.apply_angular_offset(dof, self.np_random.uniform(-err_rad, err_rad))

    def randomize_fall(self, target: bool = False):
        # Decide if we will use the target
        my_target = np.copy(self.options["desired_state"])
        if target is False:
            target = self.np_random.random() < self.options["reset_final_p"]

        # Selecting a random configuration
        initial_q = self.np_random.uniform(low=-np.pi, high=np.pi, size=(len(self.dofs),))

        # If target, we will use the q_target
        if target:
            initial_q = my_target[: len(self.dofs)]
            offset = self.np_random.uniform(-0.1, 0.1)
            initial_q[2] -= offset
            initial_q[3] += offset * 2
            initial_q[4] -= offset

        initial_q = np.clip(initial_q, self.range_low, self.range_high)

        # Set the robot initial pose
        self.apply_control(initial_q, True)
        self.sim.step()

        initial_tilt = self.np_random.uniform(-np.pi / 2, np.pi / 2)
        if target:
            initial_tilt = my_target[-1]
        T_world_trunk = tf.rotation_matrix(initial_tilt, [0, 1, 0])
        T_world_trunk[:3, 3] = [0, 0, 0.4]
        self.sim.set_T_world_site("trunk", T_world_trunk)

        # Wait for the robot to stabilize
        for _ in range(round(self.options["stabilization_time"] / self.sim.dt)):
            self.sim.step()

    def reset(
        self,
        seed: int = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.sim.reset()
        options = options or {}
        target = options.get("target", False)
        use_cache = options.get("use_cache", True)

        if use_cache and self.initial_config is None:
            warnings.warn(
                "use_cache=True but no initial config file could be loaded. Did you run standup_generate_initial.py?"
            )

        # Initial robot configuration
        if use_cache and self.initial_config is not None:
            qpos, ctrl = random.choice(self.initial_config)
            self.sim.data.qpos = qpos
            self.sim.data.ctrl = ctrl
            self.sim.data.qvel *= 0
            self.sim.step()
        else:
            self.randomize_fall(target)

        # Randomization
        self.apply_randomization()

        self.sim.reset_render()

        # Taking arms a little bit inside
        self.sim.set_control("left_shoulder_roll", -np.deg2rad(5))
        self.sim.set_control("right_shoulder_roll", np.deg2rad(5))

        # Initializing the history
        q = [self.sim.get_q(f"left_{dof}") for dof in self.dofs]
        self.q_history = [q] * self.q_history_size

        # Initializing the tilt history
        self.tilt_history = [self.get_tilt()] * self.tilt_history_size
        self.dtilt_history = [0] * self.dtilt_history_size

        # Initializing the previous action
        self.previous_actions = [np.zeros(len(self.dofs))] * self.options["previous_actions"]

        self.centroidal_force = self.sim.centroidal_force()

        return self.get_observation(), {}

    def render(self):
        self.render_mode = "human"
