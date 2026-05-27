from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp
from mujoco import mjx
from mujoco.mjx import Model, Data

from cip_rl.envs.core import Env, make_model, make_step

def get_path() -> Path:
    return Path(__file__).parent.resolve() / 'assets/double_pendulum.xml'


def make_reset(mjx_model: Model) -> callable:

    default_data = mjx.make_data(mjx_model)

    def reset(key) -> Data:
        qpos = jax.random.uniform(key, default_data.qpos.shape, minval = -jnp.pi, maxval = jnp.pi)
        qvel = jnp.zeros_like(default_data.qvel)
        return default_data.replace(qpos = qpos, qvel = qvel)

    # return jax.jit(reset)
    return reset


def make_observation() -> callable:

    def observation(mjx_data: Data) -> Array:
        '''
        Wrapped angles for observation
        '''
        
        return jnp.concatenate([
            jnp.sin(mjx_data.qpos),
            jnp.cos(mjx_data.qpos),
            mjx_data.qvel
        ])
    
    # return jax.jit(observation)
    return observation


def make_reward() -> callable:

    def reward(mjx_data: Data) -> float:
        theta0, theta1 = mjx_data.qpos
        return - 0.5 * (jnp.cos(theta0) + jnp.cos(theta0 + theta1))

    # return jax.jit(reward)
    return reward


def make_done(max_time: float) -> callable:
    
    def done(mjx_data: Data) -> bool:
        return mjx_data.time >= max_time
        
    # return jax.jit(done)
    return done


def make_env(max_time: float, device: int | None = None):

    mjx_model = make_model(get_path(), device)

    env = Env(
        model = mjx_model,
        reset = make_reset(mjx_model),
        observation = make_observation(),
        step = make_step(mjx_model),
        reward = make_reward(),
        done = make_done(max_time)
    )
    
    return env