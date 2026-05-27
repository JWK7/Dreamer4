from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx import Model, Data

from cip_rl.envs.core import Env, make_model, make_step


def get_path() -> Path:
    return Path(__file__).parent.resolve() / 'assets/cart_pole.xml'


def make_reset(mjx_model: Model) -> callable:

    default_data = mjx.make_data(mjx_model)

    def reset(key) -> Data:
        k1, k2 = jax.random.split(key)
        cart_x = jax.random.uniform(k1, (1,), minval=-1.0, maxval=1.0)
        pole_angle = jax.random.uniform(k2, (1,), minval=-jnp.pi, maxval=jnp.pi)
        qpos = jnp.concatenate([cart_x, pole_angle])
        qvel = jnp.zeros_like(default_data.qvel)
        return default_data.replace(qpos=qpos, qvel=qvel)

    return reset


def make_observation() -> callable:

    def observation(mjx_data: Data) -> Array:
        cart_x = mjx_data.qpos[0:1]
        pole_angle = mjx_data.qpos[1:2]
        return jnp.concatenate([
            cart_x,
            jnp.sin(pole_angle),
            jnp.cos(pole_angle),
            mjx_data.qvel,
        ])

    return observation


def make_reward() -> callable:

    def reward(mjx_data: Data) -> float:
        cart_x = mjx_data.qpos[0]
        pole_angle = mjx_data.qpos[1]
        return -jnp.cos(pole_angle) - 0.1 * cart_x ** 2

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
