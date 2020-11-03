import numpy as np
from .layer import Layer

np.seterr(all='ignore')

class Dense(Layer) :
  """
  Dense layer
  """
  def __init__(self,
               n_node: int,
               activation: str = 'sigmoid',
               input_shape: int = None) :
    """
    Create an instance of dense layer

    >>> n_node
    Number of node of the layer

    >>> activation
    Activation function: `relu`, `sigmoid`, `softmax`, `linear`, or `tanh`, default value is 'sigmoid'

    >>> input_shape
    Input shape
    """
    super().__init__(
      name='dense',
      weights=np.zeros(0),
      biases=np.zeros(n_node)
    )
    self.input = None
    self.input_shape = input_shape if not None else None
    self.activation = activation if activation is not None else 'sigmoid'
    self.n_node = n_node
    self.output = np.zeros(n_node)
    self.delta_weights = None

  def init_weights(self, input_shape) :
    """
    Initialize dense layer weights
    """
    if self.input_shape :
      self.weights = np.random.uniform(low=-.5, high=.5, size=(int(self.n_node), self.input_shape)).astype("float")
    else :
      self.weights = np.random.uniform(low=-.5, high=.5, size=(int(self.n_node), input_shape[0])).astype("float")

  def set_weights(self, weights: np.array) :
    """
    Set dense layer weights
    """
    self.weights = weights

  def set_biases(self, biases: np.array) :
    """
    Set dense layer biases
    """
    self.biases = biases

  def output_shape(self,
                   input_shape: tuple = None) :
    self.input_shape = input_shape
    return self.output.shape

  def relu(self, input) :
    """
    Apply ReLU function
    """
    input = 0 if input < 0 else input
    return input

  def sigmoid(self, input) :
    """
    Apply sigmoid function
    """
    input = 1 / (1 + np.exp(-input))
    return input

  def softmax(self, input) :
    """
    Apply softmax function
    """
    input = np.exp(input)
    return input / input.sum()

  def linear(self, input) :
    """
    Apply linear function
    """
    return input

  def tanh(self, input) :
    """
    Apply tanh function
    """
    input = np.tanh(input)
    return input

  def backward(self, error, learning_rate, momentum) :
    """
    Dense layer backward propagation
    """
    err = self.derivative_activation(error)
    if self.delta_weights is not None :
      self.delta_weights = learning_rate * self.derivative_weight(err) + momentum * self.delta_weights
    else :
      self.delta_weights = momentum * self.derivative_weight(err)
    return self.derivative_input(err)

  def derivative_activation(self, error) :
    """
    Compute derivative of activation function (d_E/d_net)
    """
    def derivative_relu(error) :
      """
      Compute derivative of ReLU activation function
      """
      for i in range(len(error)) :
        error[i] = 0 if error[i] < 0 else 1
        return error

    def derivative_sigmoid(error) :
      """
      Compute derivative of sigmoid activation function
      """
      for i in range(len(error)) :
        error[i] = error[i] * (1 - error[i])
        return error

    #TODO: derivative_softmax
    #TODO: derivative_linear
    #TODO: derivative_tanh

    if self.activation == 'relu' :
      return derivative_relu(error)
    elif self.activation == 'sigmoid' :
      return derivative_sigmoid(error)
    #TODO: softmax
    #TODO: linear
    #TODO: tanh

  def derivative_input(self, error) :
    """
    Compute derivative of dense layer's output with respect to its inputs
    """
    return np.dot(self.weights.transpose(), error)

  def derivative_weight(self, error) :
    """
    Compute derivative of dense layer's output with respect to its weights
    """
    input = np.zeros(self.input_shape[0] + 1)
    input[0] = 1
    input[1:] = self.input
    return np.dot(error, input.reshape(1, self.input_shape[0] + 1))

  def forward(self, input) :
    """
    Dense layer forward propagation
    """
    self.input = input
    temp_output = np.dot(self.weights, self.input)
    # if (np.dot(self.weights, input).shape[0] == 1) :
    #   print(temp_output + self.biases)
    if (self.activation == "relu") :
      for node in range(self.n_node) :
        self.output[node] = self.relu(temp_output[node] + self.biases[node])
    elif (self.activation == "sigmoid") :
      for node in range(self.n_node) :
        self.output[node] = self.sigmoid(temp_output[node] + self.biases[node])
    elif (self.activation == "softmax") :
      for node in range(self.n_node) :
        self.output[node] = self.softmax(temp_output[node] + self.biases[node])
    elif (self.activation == "linear") :
      for node in range(self.n_node) :
        self.output[node] = self.linear(temp_output[node] + self.biases[node])
    elif (self.activation == "tanh") :
      for node in range(self.n_node) :
        self.output[node] = self.tanh(temp_output[node] + self.biases[node])
    return self.output

  def update_weights(self) :
    """
    Update dense layer's biases and weights using negative gradient
    """
    print("before update:")
    print(self.weights[0][:5])
    # print(self.delta_weights)
    self.biases -= self.delta_weights[:, 0]
    self.weights -= self.delta_weights[:, 1:]
    print("after update:")
    print(self.weights[0][:5])
    self.delta_weights = None