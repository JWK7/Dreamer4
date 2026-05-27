from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx import Model, Data

from cip_rl.envs.core import Env, make_model, make_step


def get_path() -> Path:
    return Path(__file__).parent.resolve() / 'assets/humanoid_standup.xml'


def make_reset(mjx_model: Model) -> callable:

    c = 0.05
    default_data = mjx.make_data(mjx_model)

    def reset(key) -> Data:
        k1, k2 = jax.random.split(key, 2)
        qpos = default_data.qpos + jax.random.uniform(k1, shape = default_data.qpos.shape, minval = -c, maxval = c)
        qvel = default_data.qvel + jax.random.uniform(k2, shape = default_data.qvel.shape, minval = -c, maxval = c)
        data = default_data.replace(qpos = qpos, qvel = qvel)
        data = mjx.forward(mjx_model, data)
        return data

    return reset


def make_observation() -> callable:

    def observation(mjx_data: Data) -> Array:
        '''
        Observation following the MuJoCo HumanoidStandup convention:
          - qpos[2:]      : z-position of root + all joint angles (skip x,y translation)
          - qvel          : all generalised velocities (clipped for stability)
          - cinert        : body inertias in the CoM frame  (shape: n_body * 10)
          - cvel          : body velocities in the CoM frame (shape: n_body * 6)
          - qfrc_actuator : actuator forces                 (shape: nv)
          - cfrc_ext      : external contact forces         (shape: n_body * 6)
        '''
        return jnp.concatenate([
            mjx_data.qpos[2:],                                          # skip global x, y
            jnp.clip(mjx_data.qvel, -10.0, 10.0),
            jnp.clip(mjx_data.cinert[1:].ravel(), -5.0, 5.0),          # body inertia tensors: O(0.01-5)
            jnp.clip(mjx_data.cvel[1:].ravel(), -10.0, 10.0),          # body velocities: same scale as qvel
            jnp.clip(mjx_data.qfrc_actuator / 150.0, -1.0, 1.0),         # max gear(300) * max ctrl(0.4) = 120 → normalize to O(1)
            jnp.clip(mjx_data._impl.cfrc_ext[1:].ravel() * 1e-3, -1.0, 1.0) # contact forces O(100-1000N) → O(1)
        ])

    return observation


def make_reward(mjx_model: Model) -> callable:
    '''
    Reward for humanoid standup + sustained stability.
    '''

    dt = mjx_model.opt.timestep

    def reward(data: Data) -> float:

        uph_cost = data.qpos[2]
        quad_ctrl_cost = 0.001 * dt * jnp.square(data.ctrl).sum()
        quad_impact_cost = 0.5e-6 * dt * jnp.square(data.cfrc_ext).sum()
        quad_impact_cost = quad_impact_cost.clip(max = 10 * dt)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + dt
        return reward

    return reward


# def make_reward(mjx_model: Model) -> callable:
#     '''
#     Reward for humanoid standup + sustained stability.
#     '''

#     dt = mjx_model.opt.timestep

#     def reward(data: Data) -> float:

#         uph_cost = (data.qpos[2] - 0) / dt
#         quad_ctrl_cost = 0.01 * jnp.square(data.ctrl).sum()
#         quad_impact_cost = 0.5e-6 * jnp.square(data.cfrc_ext).sum()
#         quad_impact_cost = quad_impact_cost.clip(max = 10)
#         r = uph_cost - quad_ctrl_cost - quad_impact_cost + 1

#         return r / 35.0

#     return reward


def make_done(max_time: float) -> callable:

    def done(mjx_data: Data) -> bool:
        return mjx_data.time >= max_time

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