import os, time, random

import yaml
import jax
import jax.numpy as jnp
import optax

from fit import *
from utils.dino_weights import load_dino_vits


# load yaml from config file
cfg = yaml.safe_load(open('config.yaml', 'r'))
print(cfg)


@jax.jit
def train_step(state: TrainState, batch, opt_state):
    x, y = batch
    def loss_fn(params):
        logits, updates = state.apply_fn({
            'params': params,
            'batch_stats': state.batch_stats
        }, x, train=True, mutable=['batch_stats'], rngs={'dropout': key})
        loss = optax.softmax_cross_entropy(logits, jax.nn.one_hot(y, 10)).mean()
        loss_dict = {'loss': loss}
        return loss, (loss_dict, updates)

    (_, (loss_dict, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=updates['batch_stats'])
    _, opt_state = state.tx.update(grads, opt_state)
    return state, loss_dict, opt_state


@jax.jit
def eval_step(state: TrainState, batch):
    x, y = batch
    logits = state.apply_fn({
        'params': state.params,
        'batch_stats': state.batch_stats,
        }, x, train=False)
    acc = jnp.equal(jnp.argmax(logits, -1), y).mean()
    return acc


if __name__  == "__main__":
    # load dino weights
    state, params = load_dino_vits(cfg['pretrained'])

    # train
    state = fit(state, train_ds, test_ds,
                train_step=train_step, eval_step=None,
                num_epochs=100, eval_freq=10, log_name=cfg['model_name'], hparams=cfg)
