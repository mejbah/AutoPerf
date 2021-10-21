# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# ******************************************************************************
# Copyright (c) 2018 Mejbah ul Alam, Justin Gottschlich, Abdullah Muzahid
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ******************************************************************************
"""keras_autoencoder.py

A keras implemation of an autoencoder for AutoPerf, accompanied by required
callback functions.
"""

import os
import sys
import logging
import warnings
from typing import Tuple
from copy import deepcopy
from datetime import datetime

import numpy as np
from tqdm.keras import TqdmCallback

# Keras is included as part of TensorFlow in v2
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorboard.plugins import projector

from autoperf import utils
from autoperf.config import cfg

log = logging.getLogger("rich")

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")


class LossHistory(callbacks.Callback):
  """A callback class for logging the training and validation losses."""

  monitor = dict()
  record = dict()

  def __init__(self, monitor: dict = None):
    """Constructor to specify some monitoring criteria.

    Args:
        monitor (optional): Types of values to monitor at different points. Ex:
                            {'batch_end': ['loss'], 'epoch_end': ['val_loss']}
    """
    self.monitor = monitor

  def on_train_begin(self, logs: dict = None):
    """Callback that runs when training is begun."""
    for k in self.monitor.keys():
      self.record[k] = dict()
      for val in self.monitor[k]:
        self.record[k][val] = []

  def on_batch_end(self, batch: int, logs: dict = None):
    """Callback that runs at the end of every training batch.

    Args:
        batch: Training batch number.
        logs: Log dictionary containing metrics.
    """
    status = 'batch_end'
    if (logs is not None) and (self.monitor is not None) and (status in self.monitor):
      for v in self.monitor[status]:
        if v in logs:
          self.record[status][v].append(logs.get(v))

  def on_epoch_end(self, epoch: int, logs: dict = None):
    """Callback that runs at the end of every training epoch.

    Args:
        epoch: Training epoch number.
        logs: Log dictionary containing metrics.
    """
    status = 'epoch_end'
    if (logs is not None) and (self.monitor is not None) and (status in self.monitor):
      for v in self.monitor[status]:
        if v in logs:
          self.record[status][v].append(logs.get(v))


class EarlyStoppingByLossVal(callbacks.Callback):
  """A callback class for early stopping during model training."""

  def __init__(self, monitor: str = 'val_loss', value: float = 0.00001, verbose: bool = False):
    """Constructor to specify some monitoring criteria.

    Args:
        monitor (optional): Type of loss value to monitor.
        value (optional): Early stopping threshold.
        verbose (optional): Verbosity flag.
    """
    super(callbacks.Callback, self).__init__()
    self.monitor = monitor
    self.value = value
    self.verbose = verbose

  def on_epoch_end(self, epoch: int, logs: dict = None):
    """Callback at the end of an epoch, check if we should early stop.

    Args:
        epoch: Current epoch iteration.
        logs: Logs being tracked by the LossHistory class.
    """
    if logs is not None:
      current = logs.get(self.monitor)

      if current is None:
        warnings.warn(f"Early stopping requires {self.monitor} available!", RuntimeWarning)

      elif current < self.value:
        if self.verbose:
          log.info('Epoch %d: early stopping THR', epoch)
        self.model.stop_training = True


def trainAutoencoder(autoencoder: Model, trainingData: np.ndarray,
                     monitor: dict) -> Tuple[Model, list, list]:
  """Trains an autoencoder with provided trainingData.

  Args:
      autoencoder: A Keras autoencoder.
      trainingData: Data to train on.
      monitor: Data to record during training. Ex:
               {'batch_end': ['loss'], 'epoch_end': ['val_loss']}

  Returns:
      A trained model.
      The history of the monitored values.
  """
  history = LossHistory(monitor)

  # Force the training data to be evenly divisible by the batch size
  log.info('Training data length: %d', len(trainingData))

  if len(trainingData) < cfg.training.batch_size:
    log.error('Size of dataset (%d) is less than batch size (%s).',
              len(trainingData), cfg.training.batch_size)
    sys.exit(-1)

  num_batches = len(trainingData) // cfg.training.batch_size
  trainingData = trainingData[:(num_batches * cfg.training.batch_size)]

  noisyTrainingData = deepcopy(trainingData)
  noisyTrainingData += np.random.normal(loc=0.0,
                                        scale=cfg.training.noise,
                                        size=trainingData.shape)

  log.info('Noisy training data: [%s]', ', '.join(map(str, noisyTrainingData[0])))
  log.info('Original training data: [%s]', ', '.join(map(str, trainingData[0])))

  log_dir = utils.getAutoperfDir(f'logs/fit/{timestamp}')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=100, histogram_freq=1)

  autoencoder.fit(x=noisyTrainingData, y=trainingData,
                  epochs=cfg.training.epochs,
                  batch_size=cfg.training.batch_size,
                  shuffle=True,
                  validation_split=0.2,
                  verbose=0,
                  callbacks=[history, tensorboard_callback,
                             TqdmCallback(verbose=2,
                                          epochs=cfg.training.epochs,
                                          data_size=noisyTrainingData.shape[0],
                                          batch_size=cfg.training.batch_size)])

  return autoencoder, history


def predict(autoencoder: Model, test_data: np.ndarray) -> np.ndarray:
  """Runs inference with the autoencoder on `test_data`.

  Args:
      autoencoder: A Keras autoencoder.
      test_data: Data to test on.

  Returns:
      The autoencoder's predictions.
  """
  decoded_data = autoencoder.predict(test_data, verbose=False)
  return decoded_data


def getAutoencoderShape(model: Model) -> list:
  """Get the topology of an autoencoder network.

  Args:
      model: A Keras autoencoder.

  Returns:
      The network topology as a list.
  """
  topology = []
  for x in model.layers:
    if isinstance(x.output_shape, list):
      topology.append(x.output_shape[0][1])
    else:
      topology.append(x.output_shape[1])
  return topology


def getAutoencoder(input_dim: int, encoder_dim: int, layer_dims: list = None) -> Model:
  """Construct a palindromic autoencoder based on the provided configurations.

     This function creates a `tied weights` autoencoder:
 [input_dim -> layers_dims -> encoder_dim -> reverse(layers_dims) -> input_dim]

  Args:
      input_dim: Input vector shape.
      encoder_dim: Latent space shape.
      layer_dims (optional): Specific layer dimensions to use.

  Returns:
      An untrained autoencoder model.
  """
  input_layer = Input(shape=input_dim, name='input')

  index = 0
  prev = input_layer
  encoded = None
  if layer_dims is not None or len(layer_dims) == 0:
    for curr_dim in layer_dims:
      encoded = Dense(curr_dim, activation=cfg.model.activation, name=f'down_{index}')(prev)
      index += 1
      prev = encoded

  index -= 1
  encoded = Dense(encoder_dim, activation=cfg.model.activation, name='latent')(prev)
  prev = encoded
  decoded = None
  if layer_dims is not None:
    for curr_dim in reversed(layer_dims):
      decoded = Dense(curr_dim, activation=cfg.model.activation, name=f'up_{index}')(prev)
      index -= 1
      prev = decoded

  decoded = Dense(input_dim, activation='sigmoid', name='output')(prev)
  assert decoded is not None

  autoencoder = Model(input_layer, decoded)

  opt = optimizers.get(cfg.training.optimizer)
  if opt.learning_rate:
    opt.learning_rate = cfg.training.learning_rate

  autoencoder.compile(optimizer=opt, loss=cfg.training.loss)
  return autoencoder


def loadTrainedModel() -> Model:
  """Loads a trained model from the .autoperf directory.

  Returns:
      Model: A trained Keras autoencoder.
  """
  return tf.keras.models.load_model(utils.getAutoperfDir(cfg.model.filename))


def visualizeLatentSpace(model: Model, nominal_data: np.ndarray,
                         anomalous_data: np.ndarray):
  """Visualize the latent space using Tensorboard's Embeddings Projector.

  Args:
      model: A trained Keras autoencoder.
      nominal_data: A list of nominal HPC measurements.
      anomalous_data: A list of anomalous HPC measurements.
  """
  # downsample the data if needed
  desired_size = 2000
  if nominal_data.shape[0] > desired_size:
    indices = np.arange(nominal_data.shape[0])
    nominal_data = nominal_data[np.random.choice(indices, desired_size, replace=False)]

  if anomalous_data.shape[0] > desired_size:
    indices = np.arange(anomalous_data.shape[0])
    anomalous_data = anomalous_data[np.random.choice(indices, desired_size, replace=False)]

  latent = Model(inputs=model.input, outputs=model.get_layer('latent').output)
  nom_embeddings = latent.predict(nominal_data)
  anom_embeddings = latent.predict(anomalous_data)
  embeddings = tf.Variable(np.vstack([nom_embeddings, anom_embeddings]), name='embedding')
  log.info('Embeddings shape: (%s)', ', '.join(map(str, embeddings.shape)))

  log_dir = utils.getAutoperfDir(f'logs/fit/{timestamp}')
  os.makedirs(log_dir, exist_ok=True)

  with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as f:
    for _ in range(nom_embeddings.shape[0]):
      f.write('Nominal\n')
    for _ in range(anom_embeddings.shape[0]):
      f.write('Anomalous\n')

  checkpoint = tf.train.Checkpoint(embedding=embeddings)
  checkpoint.save(os.path.join(log_dir, 'embeddings.ckpt'))

  config = projector.ProjectorConfig()
  embedding = config.embeddings.add()
  embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
  embedding.metadata_path = 'metadata.tsv'
  projector.visualize_embeddings(log_dir, config)
