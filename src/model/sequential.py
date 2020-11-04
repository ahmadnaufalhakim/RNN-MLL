import json
import numpy as np
import os
import sys
import time
import math

from .layers.dense import Dense
from .layers.rnn import SimpleRNN

class SequentialModel:
  def __init__(self,
               layers: list = None):
    """
    Create a basic model instance

    >>> layers
    An array of layers
    """
    self.layers = []
    self.output_shapes = []
    self.errors = []

    if layers :
      for layer in layers:
        self.add(layer)

  def add(self,
          layer,
          initialize_weights: bool = True) :
    """
    Add layer to model

    >>> layer
    A layer instance (Convolutional, Dense, Flatten, or Pooling)

    >>> initialize_weights
    Boolean value to initialize weights randomly
    `True` if you want to automatically initialize layer weights with random values
    `False` if you want to set layer weights manually
    """
    weighted_layers = ['dense', 'recurrent']
    if len(self.layers) :
      if layer.name in weighted_layers and initialize_weights :
        layer.init_weights(self.output_shapes[-1])
      self.output_shapes.append(layer.output_shape(self.output_shapes[-1]))
    else :
      if layer.name in weighted_layers and initialize_weights :
        layer.init_weights(layer.input_shape)
      self.output_shapes.append(layer.output_shape())

    self.layers.append(layer)

  def forward(self, input: np.array) :
    """
    Forward propagation
    """
    output = input
    for layer in self.layers :
      output = layer.forward(output)
    return output

  def backward(self, outputs, targets, learning_rate=0.1, momentum=0.1) :
    """
    Backward propagation
    """
    error = None
    for i in range(len(self.layers)-1, 0, -1) :
      if (i == len(self.layers)-1) :
        error = self.calculate_error(np.array([outputs]), np.array([targets]))
      error = self.layers[i].backward(error, learning_rate, momentum)

  def update_weights(self) :
    """
    Update all layers' weights
    """
    for layer in self.layers :
      try:
        layer.update_weights()
      except Exception as e :
        pass

  def make_batches(self,
                   X: np.array,
                   y: np.array,
                   batch_size: int) :
    """
    Make batches of data based on batch size
    """
    mini_batch = math.ceil(len(X)/batch_size)
    shuffled_idx = np.array(range(len(X)))
    np.random.shuffle(shuffled_idx)

    X_batches = []
    y_batches = []

    for batch in range(mini_batch):
      i = 0
      mini_X = []
      mini_y = []
      while ((batch * batch_size + i) < ((batch + 1) * batch_size)) and ((batch * batch_size + i) < len(shuffled_idx)):
        mini_X.append(X[shuffled_idx[batch * batch_size + i]])
        mini_y.append(y[shuffled_idx[batch * batch_size + i]])
        i = i + 1
      X_batches.append(mini_X)
      y_batches.append(mini_y)

    return np.array(X_batches), np.array(y_batches)       

  def calculate_error(self,
                      output: int,
                      target: int) :
    """
    Calculate error of output
    """
    return (target - output) ** 2 / 2

  def fit(self,
          X: np.array,
          y: np.array,
          learning_rate: float = 0.1,
          momentum: float = 0.1,
          batch_size: int = 2,
          epochs: int = 2) :
    """
    Fitting training data to model
    """
    X_batches, y_batches = self.make_batches(X, y, batch_size)
    print("Fitting ...")
    for epoch in range(epochs):
      print("\n-- Epoch #{}".format(epoch + 1))
      for idx_batch, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
        print("---- Batch #{}".format(idx_batch + 1))
        for idx, (input_data, target_label) in enumerate(zip(X_batch, y_batch)) :
          print("\n------ Forward passing data #{} ...\t".format((idx + 1) + (idx_batch * batch_size)), end='')
          sys.stdout.flush()
          forward_start = time.time()
          output = self.forward(input_data)
          forward_finish = time.time()
          print("| {} s".format(forward_finish - forward_start))

          print("------ Backward passing data #{} ...\t".format((idx + 1) + (idx_batch * batch_size)), end='')
          sys.stdout.flush()
          backward_start = time.time()
          self.backward(output, target_label, learning_rate, momentum)
          backward_finish = time.time()
          print("| {} s".format(backward_finish - backward_start))
        self.update_weights()

  def predict(self,
              X: np.array,
              y: np.array = None,
              verbose: bool = False) -> list :
    """
    Predict class from an array of inputs.
    Returns an array of predicted classes
    """
    predicted_labels = []
    print('Found', len(X), 'data to be predicted', end='\n\n')

    if self.layers[-1].name == 'dense' :
      if self.layers[-1].activation == 'relu' :
        for idx, input_data in enumerate(X) :
          print('Predicting data #' + str(idx + 1), '...', end='\n' if verbose else ' ')
          sys.stdout.flush()
          start = time.time()
          output = self.forward(input_data)
          finish = time.time()
          result = np.where(output == np.max(output))
          label = result[0][0] if len(result[0]) == 1 else result[0][np.random.randint(len(result[0]))]
          predicted_labels.append(label)
          if verbose :
            print('\nModel prediction:', label, end='\t| ')
            if y is not None :
              print('Correct label:', y[idx], end='\t| ')
            print('Raw output:', output)
            print('Model finished in:', finish-start, 'seconds', end='\n\n')
          else :
            print('done')

      elif self.layers[-1].activation == 'sigmoid' :
        for idx, input_data in enumerate(X) :
          print('Predicting data #' + str(idx + 1), '...', end='\n' if verbose else ' ')
          sys.stdout.flush()
          start = time.time()
          output = self.forward(input_data)
          finish = time.time()
          label = 0 if output <= 0.5 else 1
          predicted_labels.append(label)
          if verbose :
            print('Model prediction:', label, end='\t| ')
            if y is not None :
              print('Correct label:', y[idx], end='\t| ')
            print('Raw output:', output)
            print('Model finished in:', finish-start, 'seconds', end='\n\n')
          else :
            print('done')

      else :
        raise Exception('Invalid activation function: ' + self.layers[-1].activation)

    else :
      raise Exception('Invalid layer type: ' + self.layers[-1].name)

    return predicted_labels

  def forward_propagation(self,
          X: np.array,
          y: np.array,
          learning_rate: float = 0.1,
          momentum: float = 0.1,
          batch_size: int = 1,
          epochs: int = 1) :
    """
    Fitting training data to model
    """
    X_batches, y_batches = self.make_batches(X, y, batch_size)
    outputs = []
    for epoch in range(epochs):
      for idx_batch, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
        for idx, (input_data, target_label) in enumerate(zip(X_batch, y_batch)) :
          print("\n------ Forward passing data #{} ...\t".format((idx + 1) + (idx_batch * batch_size)), end='')
          sys.stdout.flush()
          output = self.forward(input_data)
          outputs.append(list(output))
    return outputs

  def save_model_as_json(self, filename: str) :
    """
    Save sequential model to external JSON file
    """
    def default(object) :
      """
      Serializing Numpy array by converting it to a list
      """
      if isinstance(object, np.ndarray) :
        return object.tolist()

    model = {'layers': []}
    for layer in self.layers :
      layer_data = {}
      layer_data['name'] = layer.name

      if layer_data['name'] == 'conv' :
        layer_data['n_filters'] = layer.n_filters
        layer_data['filter_dim'] = layer.filter_dim
        layer_data['input_shape'] = layer.input_shape
        layer_data['padding'] = layer.padding
        layer_data['stride'] = layer.stride
        layer_data['weights'] = layer.weights
        layer_data['biases'] = layer.biases

      elif layer_data['name'] == 'dense' :
        layer_data['n_node'] = layer.n_node
        layer_data['activation'] = layer.activation
        layer_data['input_shape'] = layer.input_shape
        layer_data['weights'] = layer.weights
        layer_data['biases'] = layer.biases

      elif layer_data['name'] == 'pool' :
        layer_data['filter_dim'] = layer.filter_dim
        layer_data['stride'] = layer.stride
        layer_data['mode'] = layer.mode

      model['layers'].append(layer_data)

    if not os.path.exists('../model') :
      os.mkdir('../model')

    with open(os.path.join('../model/', filename), 'w') as output_file :
      json.dump(model, output_file, indent=4, default=default)

  def load_model_from_json(self, filename: str) :
    """
    Load sequential model from an external JSON file
    """
    self.layers = []
    self.output_shapes = []

    with open(os.path.join('../model/', filename), 'r') as input_file :
      model = json.load(input_file)
      for layer in model['layers'] :
        new_layer = None
        if layer['name'] == 'conv' :
          new_layer = Convolutional(layer['n_filters'],
                                    tuple(layer['filter_dim']),
                                    tuple(layer['input_shape']) if layer['input_shape'] is not None else None,
                                    layer['padding'],
                                    layer['stride'])
          new_layer.set_weights(np.array(layer['weights']))
          new_layer.set_biases(np.array(layer['biases']))
        elif layer['name'] == 'dense' :
          new_layer = Dense(layer['n_node'],
                            layer['activation'],
                            tuple(layer['input_shape']) if layer['input_shape'] is not None else None)
          new_layer.set_weights(np.array(layer['weights']))
          new_layer.set_biases(np.array(layer['biases']))
        elif layer['name'] == 'flatten' :
          new_layer = Flatten()
        elif layer['name'] == 'pool' :
          new_layer = Pooling(layer['filter_dim'],
                              layer['stride'],
                              layer['mode'])
        else :
          raise Exception('Invalid layer type: ' + layer['name'])

        self.add(new_layer, False)