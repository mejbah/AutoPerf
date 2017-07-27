from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import keras.callbacks
import numpy as np
from keras import optimizers

"""
help : https://keras.io/callbacks/
"""
class LossHistory(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    self.losses = []
    self.val_losses = []
  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
  def on_epoch_end(self, epoch, logs={}):
    self.val_losses.append(logs.get('val_loss'))



def trainAutoencoder( autoencoder, training_data ):
  history = LossHistory()

  #TODO: add noise??
  noise_factor = 0.1 #0.2
  noisy_training_data = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=training_data.shape)
  autoencoder.fit(noisy_training_data, training_data,
                  epochs=20, #50,
                  batch_size=100,
                  shuffle=True,
                  validation_split=0.3,
                  #validation_data=(validation_data, validation_data), 
                  verbose=1,
                  callbacks=[history])
    
  
  return autoencoder, history.losses, history.val_losses


def predict( autoencoder, test_data ):
  decoded_data = autoencoder.predict(test_data)
  return decoded_data

"""
tied weights (palindromic) autencoder : input_dim -> layers_dims -> encoder_dim -> reverse(layers_dims) -> input_dim
"""
def getAutoencoder( input_dim, encoder_dim, layer_dims=None ):
  input_layer = Input(shape=(input_dim,))
  prev = input_layer
  encoded = None
  if layer_dims != None :
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

  
#if __name__ == "__main__":
#  autencoderMain()
