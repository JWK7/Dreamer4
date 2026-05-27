from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx import Model, Data

from cip_rl.envs.core import Env, make_model, make_step


def get_path() -> Path:
    return Path(__file__).parent.resolve() / 'assets/triple_pendulum.xml'


def make_reset(mjx_model: Model) -> callable:

    default_data = mjx.make_data(mjx_model)

    def reset(key) -> Data:
        qpos = jax.random.uniform(key, default_data.qpos.shape, minval=-jnp.pi, maxval=jnp.pi)
        qvel = jnp.zeros_like(default_data.qvel)
        return default_data.replace(qpos=qpos, qvel=qvel)

    return reset


def make_observation() -> callable:

    def observation(mjx_data: Data) -> Array:
        return jnp.concatenate([
            jnp.sin(mjx_data.qpos),
            jnp.cos(mjx_data.qpos),
            mjx_data.qvel,
        ])

    return observation


def make_reward() -> callable:

    def reward(mjx_data: Data) -> float:
        # joints ordered: foot_joint, leg_joint, thigh_joint
        q0, q1, q2 = mjx_data.qpos
        # segments go upward at q=0; reward maximized when all links are upright
        return (jnp.cos(q0) + jnp.cos(q0 + q1) + jnp.cos(q0 + q1 + q2)) / 3.0

    return reward


def make_done(max_time: float) -> callable:

    def done(mjx_data: Data) -> bool:
        return mjx_data.time >= max_time

    return done


def make_env(max_time: float, device: int | None = None):

    mjx_model = make_model(get_path(), device)

    env = Env(
        model=mjx_model,
        reset=make_reset(mjx_model),
        observation=make_observation(),
        step=make_step(mjx_model),
        reward=make_reward(),
        done=make_done(max_time),
    )

    return env