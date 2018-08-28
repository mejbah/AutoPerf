from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import keras.callbacks
import numpy as np
from keras import optimizers
import configs
import math
from keras.models import model_from_yaml
"""
help : 
https://keras.io/callbacks/
https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/

"""

class LossHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.losses = []
    self.val_losses = []
  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
  def on_epoch_end(self, epoch, logs={}):
    self.val_losses.append(logs.get('val_loss'))


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
    

def trainAutoencoder( autoencoder, training_data ):
  history = LossHistory()

  #TODO: add noise??
  noise_factor = 1#0.1 #0.2
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


def predict( autoencoder, test_data ):
  decoded_data = autoencoder.predict(test_data)
  return decoded_data


def getAutoendoerShape( model ):
  topology = [ x.output_shape[1] for x in model.layers]
  return topology

"""
tied weights (palindromic) autencoder : input_dim -> layers_dims -> encoder_dim -> reverse(layers_dims) -> input_dim
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

"""
save models
"""
def save_model(model, model_name, save_dir=None):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    model_file = model_name + ".model"
    model_weights_file = model_name + ".weights"

    if save_dir != None:
      model_file = save_dir + "/" + model_file
      model_weights_file = save_dir + "/" + model_weights_file
    with open(model_file, "w") as yaml_file:
            yaml_file.write(model_yaml)
            # serialize weights to HDF5
            model.save_weights(model_weights_file)
            print("Saved model to disk: ", model_file)
 



  
#if __name__ == "__main__":
#  autencoderMain()
