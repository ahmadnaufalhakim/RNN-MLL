import numpy as np

class Layer :
  """
  Base layer class
  """
  def __init__(self,
               name = None,
               weights = None,
               biases = None) :
    self.name = name if name is not None else None
    self.weights = weights if weights is not None else None
    self.biases = biases if biases is not None else None

  def forward(self, input) -> np.array:
    """
    Forward propagation
    """
    pass