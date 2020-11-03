import numpy as np
import os
import time
# import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from model.sequential import SequentialModel
from model.layers.layer import Layer
from model.layers.dense import Dense
from model.layers.rnn import SimpleRNN

np.random.seed(13517)

if __name__ == "__main__" :
  # Prepare dataset

  # Define models
  # model1 = SequentialModel([
  #   SimpleRNN(50, (50, 1), random_mode=0),
  #   Dense(1, 'linear')
  # ])
  model1 = SequentialModel([
    SimpleRNN(64, (50, 1), return_sequences=True),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(16),
    Dense(8, 'tanh'),
    Dense(1, 'linear')
  ])
  print(model1.output_shapes)