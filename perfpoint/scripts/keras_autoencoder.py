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


def autencoderMain():
  # this is the size of our encoded representations
  encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
  layer = [784, 128, 64]
  autoencoder = getAutoencoder( 784, encoding_dim, layer )
  # this is our input placeholder
  #input_img = Input(shape=(784,))
  ## "encoded" is the encoded representation of the input
  #encoded = Dense(encoding_dim, activation='relu')(input_img)
  ## "decoded" is the lossy reconstruction of the input
  #decoded = Dense(784, activation='sigmoid')(encoded)
  #
  ## this model maps an input to its reconstruction
  #autoencoder = Model(input_img, decoded)
  #
  ## this model maps an input to its encoded representation
  #encoder = Model(input_img, encoded)
  ## create a placeholder for an encoded (32-dimensional) input
  #encoded_input = Input(shape=(encoding_dim,))
  ## retrieve the last layer of the autoencoder model
  #decoder_layer = autoencoder.layers[-1]
  ## create the decoder model
  #decoder = Model(encoded_input, decoder_layer(encoded_input))
  
      
  (x_train, _), (x_test, _) = mnist.load_data()
  
  x_train = x_train.astype('float32') / 255.
  x_test = x_test.astype('float32') / 255.
  x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
  x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
  print x_train.shape
  print x_test.shape
  trained_autoencoder, losses, val_losses = trainAutoencoder(autoencoder, x_train[:21000] )
  for loss in val_losses:
    print loss

  final_autoencoder, losses, val_losses = trainAutoencoder(trained_autoencoder, x_train[21000:] )
  #for loss in losses:
  #  print loss

  for loss in val_losses:
    print loss
  #print len(losses), len(val_losses)
  #print np.mean(losses), np.mean(val_losses)

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

  
if __name__ == "__main__":
  autencoderMain()
