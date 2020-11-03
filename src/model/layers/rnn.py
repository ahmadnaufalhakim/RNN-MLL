import numpy as np
from .layer import Layer

class SimpleRNN(Layer) :
  """
  Simple RNN layer
  """
  def __init__(self,
               n_unit: int,
               input_shape: tuple = None,
               return_sequences: bool = False,
               random_mode: int = 2) :
    """
    Create an instance of simple RNN layer

    >>> n_unit
    Number of simple RNN layer's units (hidden states)

    >>> input_shape
    Input shape

    >>> return_sequences
    Whether to return the last output in the output sequence (`False`), or the full sequence (`True`)
    Default value is False

    >>> random_mode
    Random mode of simple RNN
    `0` to initialize all weights' values with 0
    `1` to initialize all weights' values with 1
    `2` to initialize all weights' values with random values
    Default value is `2`
    """
    U = W = None
    if input_shape is not None :
      if random_mode == 0 :
        U = np.zeros((n_unit, input_shape[1]))
        W = np.zeros((n_unit, n_unit))
      elif random_mode == 1 :
        U = np.full((n_unit, input_shape[1]), 1).astype("float")
        W = np.full((n_unit, n_unit), 1).astype("float")
      else :
        U = np.random.uniform(low=-np.sqrt(1. / input_shape[1]), high=np.sqrt(1. / input_shape[1]), size=(n_unit, input_shape[1])).astype("float")
        W = np.random.uniform(low=-np.sqrt(1. / n_unit), high=np.sqrt(1. / n_unit), size=(n_unit, n_unit)).astype("float")

    super().__init__(
      name='recurrent',
      weights={
        'U': U,
        'W': W
      },
      biases=np.random.uniform(low=0, high=1, size=(n_unit, 1)).astype("float")
    )
    self.input = None
    self.n_unit = n_unit
    self.input_shape = input_shape if input_shape is not None else None
    self.return_sequences = return_sequences
    self.random_mode = random_mode
    self.output = np.zeros((input_shape[0], n_unit)) if return_sequences and input_shape is not None else np.zeros(n_unit)

  def init_weights(self, input_shape) :
    """
    Initialize simple RNN layer weights
    """
    if self.input_shape is not None :
      if self.random_mode == 0 :
        self.weights['U'] = np.zeros((self.n_unit, self.input_shape[1])).astype("float")
        self.weights['W'] = np.zeros((self.n_unit, self.n_unit)).astype("float")
      elif self.random_mode == 1 :
        self.weights['U'] = np.full((self.n_unit, self.input_shape[1]), 1).astype("float")
        self.weights['W'] = np.full((self.n_unit, self.n_unit), 1).astype("float")
      else :
        self.weights['U'] = np.random.uniform(low=-np.sqrt(1. / self.input_shape[1]), high=np.sqrt(1. / self.input_shape[1]), size=(self.n_unit, self.input_shape[1])).astype("float")
        self.weights['W'] = np.random.uniform(low=-np.sqrt(1. / self.n_unit), high=np.sqrt(1. / self.n_unit), size=(self.n_unit, self.n_unit)).astype("float")
      if self.return_sequences :
        self.output = np.zeros((self.input_shape[0], self.n_unit))

    else :
      if self.random_mode == 0 :
        self.weights['U'] = np.zeros((self.n_unit, input_shape[1])).astype("float")
        self.weights['W'] = np.zeros((self.n_unit, self.n_unit)).astype("float")
      elif self.random_mode == 1 :
        self.weights['U'] = np.full((self.n_unit, input_shape[1]), 1).astype("float")
        self.weights['W'] = np.full((self.n_unit, self.n_unit), 1).astype("float")
      else :
        self.weights['U'] = np.random.uniform(low=-np.sqrt(1. / input_shape[1]), high=np.sqrt(1. / input_shape[1]), size=(self.n_unit, input_shape[1])).astype("float")
        self.weights['W'] = np.random.uniform(low=-np.sqrt(1. / self.n_unit), high=np.sqrt(1. / self.n_unit), size=(self.n_unit, self.n_unit)).astype("float")
      if self.return_sequences :
        self.output = np.zeros((input_shape[0], self.n_unit))

  def set_weights(self, weights: np.array) :
    """
    Set simple RNN layer weights
    """
    self.weights = weights

  def set_biases(self, biases: np.array) :
    """
    Set simple RNN layer biases
    """
    self.biases = biases

  def output_shape(self,
                   input_shape: tuple = None) :
    self.input_shape = input_shape
    return self.output.shape

  def forward(self, input) :
    """
    Simple RNN layer forward propagation
    """
    self.input = input
    timesteps = input.shape[0]
    h_t = np.zeros(self.n_unit)
    if self.return_sequences :
      for timestep in range(len(timesteps)) :
        temp_output = np.dot(self.weights['U'], input[timestep]) + np.dot(self.weights['W'], h_t) + self.biases
        h_t = np.tanh(h_t)
        self.output[timestep, :] += h_t
    else :
      for timestep in range(len(timesteps)) :
        temp_output = np.dot(self.weights['U'], input[timestep]) + np.dot(self.weights['W'], h_t) + self.biases
        h_t = np.tanh(h_t)
      self.output += h_t
    return self.output