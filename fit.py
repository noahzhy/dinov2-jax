import os, time
from typing import Any

from tqdm import tqdm
import optax
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax.training import train_state, orbax_utils
import tensorboardX as tbx


key = jax.random.PRNGKey(0)


def banner_message(message):
    msg_len = 46
    msg_len = max(len(message), msg_len) if isinstance(message, str) else max(max(len(str(msg)) for msg in message), msg_len)
    # ╔══════════════╗
    # ║ Message Here ║
    # ╚══════════════╝
    print("\33[32m╔═{:═<{width}}═╗\33[0m".format("", width=msg_len))

    if isinstance(message, str):
        print("\33[32m║ {:^{width}} ║\33[0m".format(message, width=msg_len))
    elif isinstance(message, list):
        for msg in message:
            print("\33[32m║ {:^{width}} ║\33[0m".format(str(msg), width=msg_len))
    else:
        raise ValueError("message should be str or list of str.")

    print("\33[32m╚═{:═<{width}}═╝\33[0m".format("", width=msg_len))


def lr_schedule(lr, steps_per_epoch, epochs=100, warmup=5):
    return optax.warmup_cosine_decay_schedule(
        init_value=lr / 10,
        peak_value=lr,
        end_value=1e-5,
        warmup_steps=steps_per_epoch * warmup,
        decay_steps=steps_per_epoch * (epochs - warmup),
    )
    # return optax.cosine_onecycle_schedule(
    #     peak_value=lr,
    #     transition_steps=steps_per_epoch * epochs,
    #     pct_start=0.2,
    # )


# implement TrainState
class TrainState(train_state.TrainState):
    batch_stats: Any


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


def load_ckpt(state, ckpt_dir, step=None):
    if ckpt_dir is None or not os.path.exists(ckpt_dir):
        banner_message(["No checkpoint was loaded", "Training from scratch"])
        return state

    banner_message("Loading ckpt from {}".format(ckpt_dir))

    # abs path
    ckpt_dir = os.path.abspath(ckpt_dir)
    manager = ocp.PyTreeCheckpointer()

    if step is None:
        # order the folders by the number in the folder name
        step = sorted([int(f) for f in os.listdir(ckpt_dir) if f.isdigit()])[-1]

    # add default to path
    ckpt_dir = os.path.join(ckpt_dir, str(step), "default")
    return manager.restore(ckpt_dir, item=state)


def fit(state, train_ds, test_ds,
        train_step=train_step, eval_step=None,
        num_epochs=100, eval_freq=1, log_name='default', hparams=None
    ):
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
    # checkpoint manager see https://flax.readthedocs.io/en/latest/guides/training_techniques/use_checkpointing.html#with-orbax
    options = ocp.CheckpointManagerOptions(
        create=True,
        max_to_keep=1,
        save_interval_steps=eval_freq,
    )
    manager = ocp.CheckpointManager(ckpt_path, options=options)
    # logging
    banner_message(["Start training", "Device > {}".format(", ".join([str(i) for i in jax.devices()]))])
    timestamp = time.strftime("%m%d%H%M%S")
    writer = tbx.SummaryWriter("logs/{}_{}".format(log_name, timestamp))
    opt_state = state.tx.init(state.params)
    best_acc = .0
    # start training
    for epoch in range(1, num_epochs + 1):
        pbar = tqdm(train_ds)
        for batch in pbar:
            state, loss_dict, opt_state = train_step(state, batch, opt_state)
            lr = opt_state.hyperparams['learning_rate']
            pbar.set_description(f'Epoch {epoch:3d}, lr: {lr:.7f}, loss: {loss_dict["loss"]:.4f}')

            if state.step % 10 == 0 or state.step == 1:
                writer.add_scalar('train/learning_rate', lr, state.step)
                for k, v in loss_dict.items():
                    writer.add_scalar(f'train/{k}', v, state.step)

        if eval_step is not None:
            if epoch % eval_freq == 0:
                acc = []
                for batch in test_ds:
                    a = eval_step(state, batch)
                    acc.append(a)

                acc = jnp.stack(acc).mean()

                pbar.write(f'Epoch {epoch:3d}, test acc: {acc:.4f}')
                writer.add_scalar('test/accuracy', acc, epoch)
                writer.flush()

                if acc > best_acc:
                    best_acc = acc
                    manager.save(epoch, args=ocp.args.StandardSave(state), metrics={'accuracy': acc})

        else:
            manager.save(epoch, args=ocp.args.StandardSave(state))

    manager.wait_until_finished()
    banner_message(["Training finished", f"Best test acc: {best_acc:.6f}"])
    if hparams is not None:
        writer.add_hparams(hparams, {'metric/accuracy': best_acc}, name='hparam')
    writer.close()


if __name__ == "__main__":
    from model import Model
    import tensorflow_datasets as tfds


    def get_train_batches(batch_size=256):
        ds = tfds.load(name='mnist', split='train', as_supervised=True, shuffle_files=True)
        ds = ds.batch(batch_size).prefetch(1)
        return tfds.as_numpy(ds)


    def get_test_batches(batch_size=256):
        ds = tfds.load(name='mnist', split='test', as_supervised=True, shuffle_files=False)
        ds = ds.batch(batch_size).prefetch(1)
        return tfds.as_numpy(ds)


    config = {
        'lr': 5e-3,
        'batch_size': 128,
        'num_epochs': 10,
        'warmup': 3,
    }

    train_ds = get_train_batches(batch_size=config['batch_size'])
    test_ds = get_test_batches(batch_size=config['batch_size'])
    lr_fn = lr_schedule(lr=config['lr'], steps_per_epoch=len(train_ds), epochs=config['num_epochs'], warmup=config['warmup'])

    key = jax.random.PRNGKey(0)
    model = Model()
    x = jnp.ones((1, 28, 28, 1))
    var = model.init(key, x, train=True)

    state = TrainState.create(
        apply_fn=model.apply,
        params=var['params'],
        batch_stats=var['batch_stats'],
        tx=optax.inject_hyperparams(optax.adam)(lr_fn),
    )

    import time
    start = time.perf_counter()

    fit(state, train_ds, test_ds,
        train_step=train_step,
        eval_step=eval_step,
        eval_freq=1,
        num_epochs=config['num_epochs'],
        hparams=config,
        log_name='mnist')

    print("Elapsed time: {} ms".format((time.perf_counter() - start) * 1000))

    state = load_ckpt(state, "./checkpoints")

    acc = []
    for batch in test_ds:
        a = eval_step(state, batch)
        acc.append(a)
    acc = jnp.stack(acc).mean()

    print("\33[32mEval Accuracy: {:.6f}\33[0m".format(acc))
