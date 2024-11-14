# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Callable, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import specs
from brax.training.types import Params
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1

##### import custom brax functions
import custom_brax.masked_running_statistics as running_statistics
from custom_brax.masked_running_statistics import RunningStatisticsState
import custom_brax.custom_losses as ppo_losses
from custom_brax import custom_ppo_networks
from custom_brax import custom_wrappers

import flax
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from etils import epath
from pathlib import Path

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = "i"

'''
Update note: 
New PPO training and evaluation function that includes sensory network
PPOSensingNetworkParams is a new dataclass that includes sensory params, policy params and value params
PPOSensingImitationNetworks is a new network that includes sensory network, policy network and value network
HG on 2024-11-08
'''

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: Union[ppo_losses.PPONetworkParams, ppo_losses.PPOSensingNetworkParams]
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def train(
    environment: Union[envs_v1.Env, envs.Env],
    num_timesteps: int,
    episode_length: int,
    checkpoint_manager: ocp.CheckpointManager,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    kl_loss: bool = False,
    kl_weight: float = 1e-3,
    discounting: float = 0.9,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 1,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    deterministic_eval: bool = False,
    ###### TO DO: add make_feco_intention_ppo_networks
    network_factory: types.NetworkFactory[
        Union[custom_ppo_networks.PPOSensingImitationNetworks, custom_ppo_networks.PPOImitationNetworks]
    ] = custom_ppo_networks.make_intention_ppo_networks, # custom_ppo_networks.make_separate_sensory_and_intention_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,  # includes sensory and intention networks
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    freeze_mask_fn=None,
    checkpoint_path: Optional[Path] = None,
    ###### TO DO: add make_feco_intention_ppo_networks ----> DONE
    checkpoint_network_factory: types.NetworkFactory[
        Union[custom_ppo_networks.PPOSensingImitationNetworks, custom_ppo_networks.PPOImitationNetworks]
    ] = custom_ppo_networks.make_intention_ppo_networks,
    tracking_task_obs_size: int = 470,
    continue_training: bool = False,
    custom_wrap: bool = False,
    freeze_decoder: bool = False,
    freeze_sensory: bool = True,
):
    """PPO training.

    Args:
      environment: the environment to train
      num_timesteps: the total number of environment steps to use during training
      episode_length: the length of an environment episode
      action_repeat: the number of timesteps to repeat an action
      num_envs: the number of parallel environments to use for rollouts
        NOTE: `num_envs` must be divisible by the total number of chips since each
          chip gets `num_envs // total_number_of_chips` environments to roll out
        NOTE: `batch_size * num_minibatches` must be divisible by `num_envs` since
          data generated by `num_envs` parallel envs gets used for gradient
          updates over `num_minibatches` of data, where each minibatch has a
          leading dimension of `batch_size`
      max_devices_per_host: maximum number of chips to use per host process
      num_eval_envs: the number of envs to use for evluation. Each env will run 1
        episode, and all envs run in parallel during eval.
      learning_rate: learning rate for ppo loss
      entropy_cost: entropy reward for ppo loss, higher values increase entropy
        of the policy
      discounting: discounting rate
      seed: random seed
      unroll_length: the number of timesteps to unroll in each environment. The
        PPO loss is computed over `unroll_length` timesteps
      batch_size: the batch size for each minibatch SGD step
      num_minibatches: the number of times to run the SGD step, each with a
        different minibatch with leading dimension of `batch_size`
      num_updates_per_batch: the number of times to run the gradient update over
        all minibatches before doing a new environment rollout
      num_evals: the number of evals to run during the entire training run.
        Increasing the number of evals increases total training time
      num_resets_per_eval: the number of environment resets to run between each
        eval. The environment resets occur on the host
      normalize_observations: whether to normalize observations
      reward_scaling: float scaling for reward
      clipping_epsilon: clipping epsilon for PPO loss
      gae_lambda: General advantage estimation lambda
      deterministic_eval: whether to run the eval with a deterministic policy
      network_factory: function that generates networks for (sensory), policy and value
        functions
      progress_fn: a user-defined callback function for reporting/plotting metrics
      normalize_advantage: whether to normalize advantage estimate
      eval_env: an optional environment for eval only, defaults to `environment`
      policy_params_fn: a user-defined callback function that can be used for
        saving policy checkpoints
      randomization_fn: a user-defined callback function that generates randomized
        environments

    Returns:
      Tuple of (make_policy function, network params, metrics)
    """
    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count

    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        )
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_sensory, key_policy, key_value, policy_params_fn_key = jax.random.split(global_key, 4)
    del global_key

    assert num_envs % device_count == 0

    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(randomization_fn, rng=randomization_rng)

    if isinstance(environment, envs.Env):
        if custom_wrap:
            wrap_for_training = custom_wrappers.wrap
        else:
            wrap_for_training = envs.training.wrap
    else:
        wrap_for_training = envs_v1.wrappers.wrap_for_training

    env = wrap_for_training(
        environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    reset_fn = jax.jit(jax.vmap(env.reset))
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(key_envs, (local_devices_to_use, -1) + key_envs.shape[1:])
    env_state = reset_fn(key_envs)

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    task_obs_size = _unpmap(env_state.info["task_obs_size"])[0]
    ### pass essential args shared across all constructors
    ppo_network = network_factory(
        env_state.obs.shape[-1],
        task_obs_size,
        env.action_size,
        preprocess_observations_fn=normalize,
    )


    # ###### TO DO: add new inference function --> DONE
    if isinstance(ppo_network, custom_ppo_networks.PPOSensingImitationNetworks):
        make_policy = custom_ppo_networks.make_sensory_inference_fn(ppo_network)
    else:
        make_policy = custom_ppo_networks.make_inference_fn(ppo_network)

    
    ## Initialize network params
    if isinstance(ppo_network, custom_ppo_networks.PPOSensingImitationNetworks):
        init_params = ppo_losses.PPOSensingNetworkParams(
            sensory=ppo_network.sensory_network.init(key_sensory),  # redefined network params
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )
    ##### init for older PPO networks without sensory params
    else:
        init_params = ppo_losses.PPONetworkParams(
            policy=ppo_network.policy_network.init(key_policy),
            value=ppo_network.value_network.init(key_value),
        )

    if freeze_mask_fn is not None:
        optimizer = optax.multi_transform(
            {
                "learned": optax.adam(learning_rate=learning_rate),
                "frozen": optax.set_to_zero(),
            },
            freeze_mask_fn(init_params),
        )
        print("Freezing layers")
    else:
        optimizer = optax.adam(learning_rate=learning_rate)
        print("Not freezing any layers")

    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(
            init_params
        ),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(
            specs.Array(env_state.obs.shape[-1:], jnp.dtype("float32"))
        ),
        env_steps=0,
    )

    
    # Load from checkpoint, and set params for decoder if freeze, or all if continuing
    if checkpoint_path is not None and epath.Path(checkpoint_path).exists():
        logging.info("restoring from checkpoint %s", checkpoint_path)
        #env_steps = int(epath.Path(checkpoint_path).stem)
        eval_it = int(epath.Path(checkpoint_path).stem)
        print('reset env_steps to:', env_steps)
        ckptr = ocp.CompositeCheckpointHandler()
        tracking_task_obs_size = 935
        tracking_obs_size = (
            env_state.obs.shape[-1] - task_obs_size + tracking_task_obs_size
        )
        checkpoint_ppo_network = checkpoint_network_factory(
            tracking_obs_size,
            tracking_task_obs_size,
            env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        if isinstance(checkpoint_ppo_network, custom_ppo_networks.PPOSensingImitationNetworks):
            checkpoint_init_params = ppo_losses.PPOSensingNetworkParams(
                sensory=checkpoint_ppo_network.sensory_network.init(key_sensory),
                policy=checkpoint_ppo_network.policy_network.init(key_policy),
                value=checkpoint_ppo_network.value_network.init(key_value),
            )
        else:
            checkpoint_init_params = ppo_losses.PPONetworkParams(
                policy=checkpoint_ppo_network.policy_network.init(key_policy),
                value=checkpoint_ppo_network.value_network.init(key_value),
            )
        target = ocp.args.Composite(
            normalizer_params=ocp.args.StandardRestore(
                running_statistics.init_state(
                    specs.Array(tracking_obs_size, jnp.dtype("float32"))
                )
            ),
            params=ocp.args.StandardRestore(checkpoint_init_params),
            env_steps=ocp.args.ArrayRestore(0),
        )
        loaded_ckpt = ckptr.restore(checkpoint_path.resolve(), args=target)
        loaded_normalizer_params = loaded_ckpt["normalizer_params"]
        loaded_params = loaded_ckpt["params"]
        env_steps = loaded_ckpt["env_steps"]
        print('loaded env_steps:', env_steps)

        # Only partially replace initial policy if freezing decoder
        ##### TO DO: freeze sensory params ----> doesn't handle sensory params within decoder properly yet
        if freeze_decoder or freeze_sensory:
            running_statistics_mask = jnp.arange(env_state.obs.shape[-1]) < int(
                task_obs_size
            )
            mean = training_state.normalizer_params.mean.at[task_obs_size:].set(
                loaded_normalizer_params.mean[tracking_task_obs_size:]
            )
            std = training_state.normalizer_params.std.at[task_obs_size:].set(
                loaded_normalizer_params.std[tracking_task_obs_size:]
            )
            summed_variance = training_state.normalizer_params.summed_variance.at[
                task_obs_size:
            ].set(loaded_normalizer_params.summed_variance[tracking_task_obs_size:])
            normalizer_params = RunningStatisticsState(
                count=jnp.zeros(()), mean=mean, summed_variance=summed_variance, std=std
            )
            assert (
                running_statistics_mask.shape
                == training_state.normalizer_params.mean.shape
            )
        else:
            if isinstance(ppo_network, custom_ppo_networks.PPOSensingImitationNetworks):
                init_params = init_params.replace(sensory=loaded_params.sensory)
            init_params = init_params.replace(policy=loaded_params.policy)
            running_statistics_mask = None
            normalizer_params = loaded_normalizer_params

        if freeze_sensory:
            if isinstance(ppo_network, custom_ppo_networks.PPOSensingImitationNetworks):
                init_params.sensory["params"] = loaded_params.sensory["params"]
            for key in loaded_params.policy.keys():
                if 'sensory' in key:
                    init_params.policy["params"][key] = loaded_params.policy["params"][key]
        else:
            if isinstance(ppo_network, custom_ppo_networks.PPOSensingImitationNetworks):
                init_params = init_params.replace(sensory=loaded_params.sensory)
        
        if freeze_decoder:
            ######## Need to change this to include freezing sensory network
            init_params.policy["params"]["decoder"] = loaded_params.policy["params"]["decoder"]
        else:
            init_params = init_params.replace(policy=loaded_params.policy)

        if continue_training:
            print('continuing training')
            init_params = init_params.replace(value=loaded_params.value)
            normalizer_params=loaded_normalizer_params
        else:
            print('starting from env_steps = 0')
            env_steps = 0
            eval_it = 0

        training_state = (
            TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
                optimizer_state=optimizer.init(
                    init_params
                ),  # pytype: disable=wrong-arg-types  # numpy-scalars
                params=init_params,
                normalizer_params=normalizer_params,
                env_steps=env_steps,
            )
        )
    else:
        running_statistics_mask = None
        eval_it = 0

    if num_timesteps == 0:
        return (
            make_policy,
            (training_state.normalizer_params, training_state.params, training_state.env_steps),
            {},
        )
    
    loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        kl_loss=kl_loss,
        kl_weight=kl_weight,
        discounting=discounting,
        reward_scaling=reward_scaling,
        gae_lambda=gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
    )

    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    def minibatch_step(
        carry,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params, normalizer_params, data, key_loss, optimizer_state=optimizer_state
        )

        return (optimizer_state, params, key), metrics

    def sgd_step(
        carry,
        unused_t,
        data: types.Transition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics

    def training_step(
        carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
    ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)


        #### TO DO: add sensory params  ---> DONE
        if isinstance(ppo_network, custom_ppo_networks.PPOSensingImitationNetworks):
            policy = make_policy(
                (training_state.normalizer_params, training_state.params.policy, training_state.params.sensory)
                )
        else:
            policy = make_policy(
                (training_state.normalizer_params, training_state.params.policy)
                )

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                current_key,
                unroll_length,
                extra_fields=("truncation",),
            )
            return (next_state, next_key), data

        (state, _), data = jax.lax.scan(
            f,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        assert data.discount.shape[1:] == (unroll_length,)

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            mask=running_statistics_mask,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )

        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key), metrics

    def training_epoch(
        training_state: TrainingState, state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (
            training_state,
            env_state,
            metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    training_state = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    if not eval_env:
        eval_env = environment
    if randomization_fn is not None:
        v_randomization_fn = functools.partial(
            randomization_fn, rng=jax.random.split(eval_key, num_eval_envs)
        )
    eval_env = wrap_for_training(
        eval_env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

    evaluator = acting.Evaluator(  # expects:    
        eval_env,       # eval_env: envs.Env,
        functools.partial(make_policy, deterministic=deterministic_eval),   # eval_policy_fn: Callable[[PolicyParams], Policy], --> technically PolicyParams is only (normalizer, policy) but can send (normalzer, actor, sensory) if make_policy fn expects it and still returns a policy
        num_eval_envs=num_eval_envs,        # num_eval_envs: int,   # num_eval_envs: int,
        episode_length=episode_length,      # episode_length: int, 
        action_repeat=action_repeat,        #  action_repeat: int,
        key=eval_key,                       # key: PRNGKey,    
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        print('-------------- running initial evaluation --------------')
        if isinstance(ppo_network, custom_ppo_networks.PPOSensingImitationNetworks):
            pcopy = (training_state.normalizer_params, training_state.params.policy, training_state.params.sensory)
        else:
            pcopy = (training_state.normalizer_params, training_state.params.policy)
        metrics = evaluator.run_evaluation(
            _unpmap(pcopy),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    print('-------------- starting training --------------')
    training_metrics = {}
    training_walltime = 0
    current_step = 0
    for it in range(eval_it, num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))
            print('current step = ...... ', current_step)

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id == 0:
            # Run evals.
            if isinstance(ppo_network, custom_ppo_networks.PPOSensingImitationNetworks):
                pcopy = (training_state.normalizer_params, training_state.params.policy, training_state.params.sensory)
            else:
                pcopy =  (training_state.normalizer_params, training_state.params.policy)
            metrics = evaluator.run_evaluation(
                _unpmap(  pcopy  ),
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)
            results = _unpmap(
                (
                    training_state.normalizer_params,
                    training_state.params,
                    training_state.env_steps,
                )
            )
            # Save checkpoint
            print('saving checkpoint = ', results[2])
            checkpoint_manager.save(
                it,
                results,
                args=ocp.args.Composite(
                    normalizer_params=ocp.args.StandardSave(results[0]),
                    params=ocp.args.StandardSave(results[1]),
                    env_steps=ocp.args.ArraySave(results[2]),
                ),
            )

            _, policy_params_fn_key = jax.random.split(policy_params_fn_key)
            policy_params_fn(current_step, make_policy, results, policy_params_fn_key)

    total_steps = current_step
    assert total_steps >= num_timesteps

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap((training_state.normalizer_params, training_state.params))
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)