"""
Copyright (c) 2018 Mejbah ul Alam, Justin Gottschlich, Abdullah Muzahid

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

"""
File name: autoperf.py
File description: A keras implemation of Autoencoder for AutoPerf accompanied with classes required callback functions
"""



from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import keras.callbacks
import numpy as np
from keras import optimizers
import configs
import math

"""
A callback class for logging training and validation losses in keras model
"""
class LossHistory(keras.callbacks.Callback):

  def on_train_begin(self, logs={}):
    self.losses = []
    self.val_losses = []
  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
  def on_epoch_end(self, epoch, logs={}):
    self.val_losses.append(logs.get('val_loss'))


"""
A callback for early stopping during model training
"""
class EarlyStoppingByLossVal(keras.callbacks.Callback):

  def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
    super(Callback, self).__init__()
    self.monitor = monitor
    self.value = value
    self.verbose = verbose

  def on_epoch_end(self, epoch, logs={}):
    current = logs.get(self.monitor)
    if current is None:
      warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
    if current < self.value:
      if self.verbose > 0:
        print("Epoch %05d: early stopping THR" % epoch)
      self.model.stop_training = True

"""
Alternative to early stopping : a new variaance termination algorithm
"""
class VarianceTermination(keras.callbacks.Callback):

  def __init_(self, monitor='val_loss', verbose=0, epsilon=0.05, repeatedSuccess=0):
    self.losses = []
    self.monitor = monitor
    self.epsilon = epsilon
    self.repeatedSuccess = repeatedSuccess
    self.lastStop = 0

  def on_epoch_end(self, batch, logs={}):
    self.losses.append(log.get(self.monitor))
    for i in range(1, v.size()):
      average_var_piecewise = average_variance(self.losses, i - math.ceil(i/10.0), i);
      average_var_total = average_variance(v,0,i) / math.ceil(1 + (i/2.0))
      if average_var_piecewise < average_var_piecewise and self.lastStop >= self.repeatedSuccess and average_var_piecewise < self.epsilon:
        self.model.stop_training = True
      elif average_var_piecewise < average_var_total:
        self.lastStop += 1
      else:
        self.lastStop = 0


"""
Training an autoendoer with training_data
"""
def trainAutoencoder( autoencoder, training_data ):

  history = LossHistory()

  noise_factor = 1 #0.1 #0.2
  noisy_training_data = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=training_data.shape)
  autoencoder.fit(noisy_training_data, training_data,
                  epochs=configs.TRAINING_EPOCHS,  #20, #50,
                  batch_size=configs.TRAINING_BATCH_SIZE,
                  shuffle=True,
                  validation_split=0.3,
                  #validation_data=(validation_data, validation_data), 
                  verbose=1,
                  callbacks=[history])

  return autoencoder, history.losses, history.val_losses


"""
Inferencing of autoencoder with test_data
"""
def predict( autoencoder, test_data ):

  decoded_data = autoencoder.predict(test_data)
  return decoded_data


"""
Get the topology of an autoencoder network
"""
def getAutoendoerShape( model ):

  topology = [ x.output_shape[1] for x in model.layers]
  return topology


"""
Construct an autoencoder model based on configurations provided
This fucntion creates tied weights (palindromic) autencoder : input_dim -> layers_dims -> encoder_dim -> reverse(layers_dims) -> input_dim
"""
def getAutoencoder( input_dim, encoder_dim, layer_dims=None ):

  input_layer = Input(shape=(input_dim,))
  prev = input_layer
  encoded = None
  if layer_dims != None or len(layer_dims)==0 :
    for curr_dim in layer_dims:
      encoded = Dense( curr_dim, activation = 'sigmoid' )(prev)
      prev = encoded

  encoded = Dense( encoder_dim, activation = 'sigmoid' ) (prev)
  prev = encoded
  decoded = None
  if layer_dims != None :
    for curr_dim in reversed(layer_dims):
      decoded = Dense( curr_dim, activation = 'sigmoid' )(prev)
      prev = decoded
  decoded = Dense( input_dim, activation = 'sigmoid' )(prev)

  assert decoded != None
  autoencoder = Model(input_layer, decoded)
  #autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
  #optimizer_var = optimizers.RMSprop(lr=0.01)
  optimizer_var = optimizers.SGD(lr=0.01)
  autoencoder.compile(optimizer=optimizer_var, loss='mean_squared_error')
  return autoencoder
