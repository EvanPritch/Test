import itertools
import json
from datetime import datetime

import numpy as np
import os
import tensorflow as tf
from sacred import Experiment
from sacred.observers import MongoObserver

tf.enable_eager_execution()

ex = Experiment('rnn_particle_type', interactive=True)
ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='ether'))

layers = tf.keras.layers

_RECURRENT_LAYER_TYPE = {
    (True, 'lstm'): layers.CuDNNLSTM,
    (True, 'gru'): layers.CuDNNGRU,
    (False, 'lstm'): layers.LSTM,
    (False, 'gru'): layers.GRU
}


def build_model(rnn_units, rnn_type, batch_size, vocab_size, embedding_dim, stateful):
    """Builds an RNN with the provided specifications.
    Args:
    Returns:
        The model.
    """
    rnn = _RECURRENT_LAYER_TYPE[(tf.test.is_gpu_available(), rnn_type)]

    model = tf.keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim,
                               batch_input_shape=[batch_size, None]))
    for units in rnn_units:
        model.add(rnn(units,
                      return_sequences=True,
                      stateful=stateful))
    model.add(layers.Dense(vocab_size))

    return model


def split_input_target(chunk):
    """Given a chunk of particles, returns input and target particles, where the
    target particles is the sequence that should come after the input.
    Args:
        chunk: The chunk of particles to split up.
    Returns:
        input, target
    """
    input = chunk[:-1]
    target = chunk[1:]
    return input, target


@ex.config
def config():
    training_data = "/Users/evanpritchard/Desktop/40gev/mass/particle-type-5000000.json"
    logdir = "/Users/evanpritchard/Desktop/SummerResearch2019/Code"
    checkpoint_period = 1
    epochs = 5
    seq_length = 50
    batch_size = 32
    buffer_size = 10000
    learning_rate = 0.001
    rnn_units = (512, 512)
    rnn_type = 'lstm'
    stateful = False
    embedding_dim = 100


@ex.config_hook
def config_hook(config, command_name, logger):
    if command_name == 'train':
        if config['training_data'] is None:
            logger.error('Path to training data must be provided.')
            exit(1)
        if not config['training_data'].endswith('.json'):
            logger.error('Training data should be in a .json file.')
            exit(1)
        if config['logdir'] is None:
            logger.error('A log directory must be provided.')
            exit(1)
        if config['rnn_type'] not in ['lstm', 'gru']:
            logger.error('rnn_type must be "lstm" or "gru"')
            exit(1)

    return config


def intersperse(delimiter, seq):
    return list(itertools.islice(itertools.chain.from_iterable(zip(itertools.repeat(delimiter), seq)), 1, None))


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


@ex.automain
def train(training_data, logdir, checkpoint_period, epochs, seq_length, batch_size, buffer_size, rnn_units, rnn_type,
          learning_rate, embedding_dim, stateful, seed, _log, _run, _rnd):
    tf.set_random_seed(seed)

    logdir = os.path.join(logdir, ex.path, datetime.now().strftime('%Y%m%d%H%M%S'))
    _log.info('Logs will be saved to: {}'.format(logdir))
    os.makedirs(logdir)

    with open(training_data, 'r') as fp:
        data = json.load(fp)

    for event in data:
        event.sort()

    assert 0 not in list(itertools.chain(*data))
    data = intersperse([0], data)
    data = list(itertools.chain(*data))
    vocab = sorted(set(data))
    id2index = {u: i for i, u in enumerate(vocab)}
    data = [id2index[p] for p in data]
    dataset = tf.data.Dataset.from_tensor_slices(data)
    sequences = dataset.batch(seq_length + 1, drop_remainder=True)
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    with open(os.path.join(logdir, 'id2index.json'), 'w') as fp:
        json.dump(id2index, fp)
    _run.add_artifact(os.path.join(logdir, 'id2index.json'))

    steps_per_epoch = (len(data) // seq_length) // batch_size

    # Build and compile the model
    model = build_model(rnn_units, rnn_type, batch_size, len(vocab), embedding_dim, stateful)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()

    # Checkpoint callback for saving model weights
    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'epoch_{epoch}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                             period=checkpoint_period,
                                                             save_weights_only=True)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # TensorBoard callback
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, batch_size=batch_size, update_freq='batch')

    # Train the model
    model.fit(dataset.repeat(),
              steps_per_epoch=steps_per_epoch,
              epochs=epochs,
              verbose=1,
              callbacks=[checkpoint_callback, tb_callback])

    with open(os.path.join(logdir, 'model_info.json'), 'w') as fp:
        model_info = {
            'rnn_units': rnn_units,
            'rnn_type': rnn_type,
            'seq_length': seq_length,
            'stateful': stateful,
            'embedding_dim': embedding_dim
        }
        json.dump(model_info, fp)
