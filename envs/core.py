from dataclasses import dataclass
from typing import Optional

import chex
import mujoco
from mujoco import mjx
from mujoco.mjx import Model, Data
import jax
from jax import Array
from jax import numpy as jnp

from cip_rl.utils import split_state, get_state

def make_model(path: str, device: Optional[int] = None) -> Model:
    '''
    Loads in an mjx model.
    '''

    model = mujoco.MjModel.from_xml_path(str(path))
    mjx_model = mjx.put_model(model, device)
    return mjx_model


def make_step(model: Model):
    '''
    Makes a stepper function which is a closure over the model.
    '''

    def step(data: Data, ctrl: Array) -> Data:
        data = data.replace(ctrl = ctrl)
        return mjx.step(model, data)
    
    return step


def make_dynamics(model: Model):
    '''
    Makes a differentiable step function. The standard step function takes in the entire data object which is not efficient for jacfwd.
    This function will take in only the state vector and action vector and output a new state vector. This function is mostly equivelant to the original step function.
    '''

    data_template = mjx.make_data(model)
    nq = model.nq
    step = make_step(model)

    def dynamics(state: Array, action: Array):
        qpos, qvel = split_state(state, nq)
        data = data_template.replace(qpos = qpos, qvel = qvel, ctrl = action)
        data = step(data, action)
        return get_state(data)
    
    return dynamics


@chex.dataclass(frozen = True)
class Env:
    '''
    Holds the neccissary function for propagating the dynamical system
    '''
    model: Model
    reset: callable
    observation: callable
    step: callable
    reward: callable
    done: callable

    @property
    def dt(self):
        return self.model.opt.timestep
    
    @property
    def control_low(self) -> Array:
        return self.model.actuator_ctrlrange[:, 0]
    
    @property
    def control_high(self) -> Array:
        return self.model.actuator_ctrlrange[:, 1]
    
    @property
    def control_dim(self) -> int:
        return self.model.nu


def batchify(env: Env):

    batched_env = Env(
        model =         env.model,
        reset =         jax.vmap(env.reset),
        observation =   jax.vmap(env.observation),
        step =          jax.vmap(env.step),
        reward =        jax.vmap(env.reward),
        done =          jax.vmap(env.done)
    )

    return batched_env


def make_auto_reset(env: Env):

    def auto_reset(data: Data, done: Array, key: Array) -> Data:
        keys = jax.random.split(key, done.shape[0])
        reset_data = env.reset(keys)
        def select(d, r):
            mask = done.reshape(done.shape + (1,) * (d.ndim - 1))
            return jnp.where(mask, r, d)
        return jax.tree.map(select, data, reset_data)

    return auto_reset