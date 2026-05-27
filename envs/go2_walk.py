from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx import Model, Data

from cip_rl.envs.core import Env, make_model, make_step


def get_path() -> Path:
    return Path(__file__).parent.resolve() / 'assets/unitree_go2/scene_mjx.xml'


def make_reset(mjx_model: Model) -> callable:

    reset_qpos = jnp.array(mjx_model.key_qpos[0])  ## stores keys as (1, ...)
    reset_qvel = jnp.array(mjx_model.key_qvel[0])
    default_data = mjx.make_data(mjx_model)

    def reset(key):

        data = default_data.replace(qpos = reset_qpos, qvel = reset_qvel)
        data = mjx.forward(mjx_model, data)
        
        return data
    
    return reset


def make_observation() -> callable:
    '''
    Observation for Go2 walking:
      - qpos[2:]   : base z + orientation quat + 12 joint angles  (17 values)
      - qvel       : base linear/angular vel + 12 joint vels       (18 values)
    '''

    def observation(mjx_data: Data) -> Array:
        return jnp.concatenate([
            # mjx_data.qpos[2:],
            jnp.clip(mjx_data.qpos[2:], -10.0, 10.0),
            jnp.clip(mjx_data.qvel, -10.0, 10.0),
        ])

    return observation


def make_reward(mjx_model: Model) -> callable:
    """
    Walking reward for Go2, based on ideas from Argo-Robot/quadrupeds_locomotion.

    Key components:
      - Exponential Gaussian forward velocity tracking (sigma = 0.25)
      - Strong height penalty to maintain stance
      - Vertical velocity suppression for smooth gait
      - L1 default-pose similarity to encourage a natural stance
      - Light control cost
    """

    target_vx = 0.8
    tracking_sigma = 0.25
    nominal_height = 0.27  # from keyframe: qpos[2] = 0.27 at the home stance
    default_joints = jnp.array(mjx_model.key_qpos[0][7:])  # 12 default joint angles

    def reward(mjx_data: Data) -> float:
        vx = mjx_data.qvel[0]
        vy = mjx_data.qvel[1]
        vz = mjx_data.qvel[2]
        yaw_rate = mjx_data.qvel[5]
        z = mjx_data.qpos[2]

        # Exponential Gaussian tracking: reward = 1 at perfect match, decays smoothly
        tracking_reward = jnp.exp(-((vx - target_vx) ** 2) / tracking_sigma)


        # Strong height penalty to keep the robot in its nominal stance
        height_penalty = (z - nominal_height) ** 2

        # Suppress vertical bouncing for a smoother gait
        vz_penalty = vz ** 2

        # return jnp.clip(
        #     1.0 * tracking_reward
        #     - 20.0 * height_penalty
        #     - 1.0 * vz_penalty,
        #     -10, 5.0
        # )

        # Penalize lateral drift and spinning to enforce a fixed forward direction
        lateral_penalty = vy ** 2
        yaw_penalty = yaw_rate ** 2

        # L1 distance from default joint angles — encourages natural leg positions
        joint_pos = mjx_data.qpos[7:]
        pose_penalty = jnp.sum(jnp.abs(joint_pos - default_joints))

        # Light control cost
        ctrl_cost = jnp.square(mjx_data.ctrl).sum()

        return jnp.clip(
            1.0 * tracking_reward
            - 20.0 * height_penalty
            - 1.0 * vz_penalty
            - 0.5 * lateral_penalty
            - 0.2 * yaw_penalty
            - 0.1 * pose_penalty
            - 0.001 * ctrl_cost,
            -10.0, 5.0  # max is 1.0 by construction, min guards against explosions
        )

    return reward


def make_done(max_time: float, min_height: float = 0.20) -> callable:
 
    def done(mjx_data: Data) -> bool:
        fell      = mjx_data.qpos[2] < min_height
        timed_out = mjx_data.time >= max_time
        exploded  = jnp.logical_not(jnp.isfinite(jnp.concatenate([mjx_data.qpos, mjx_data.qvel])).all())
        return jnp.logical_or(jnp.logical_or(fell, timed_out), exploded)
 
    return done


def make_env(max_time: float, device: int | None = None):

    mjx_model = make_model(get_path(), device)

    env = Env(
        model = mjx_model,
        reset = make_reset(mjx_model),
        observation = make_observation(),
        step = make_step(mjx_model),
        reward = make_reward(mjx_model),
        done = make_done(max_time)
    )

    return env