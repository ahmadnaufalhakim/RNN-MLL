import numpy as np
import pandas as pd
import os
import time
# import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from model.sequential import SequentialModel
from model.layers.layer import Layer
from model.layers.dense import Dense
from model.layers.rnn import SimpleRNN

from sklearn.preprocessing import MinMaxScaler
  
np.random.seed(13517)

if __name__ == "__main__" :
  # Prepare dataset

  # Define models
  # model1 = SequentialModel([
  #   SimpleRNN(50, (50, 1), random_mode=0),
  #   Dense(1, 'linear')
  # ])
  # model1 = SequentialModel([
  #   SimpleRNN(64, (50, 1), return_sequences=True),
  #   SimpleRNN(32, return_sequences=True),
  #   SimpleRNN(16),
  #   Dense(8, 'tanh'),
  #   Dense(1, 'linear')
  # ])
  # print(model1.output_shapes)

  data = pd.read_csv("../data/train_IBM.csv", index_col='Date', parse_dates=['Date'])
  data.dropna(inplace=True)

  training_set = data['Close'].values.reshape(-1, 1)
  test_set = pd.read_csv("../data/sample_submission.csv", index_col='Date', parse_dates=['Date']).values.reshape(-1, 1)
  
  scaler = MinMaxScaler(feature_range=(0,1))
  training_set_scaled = scaler.fit_transform(training_set)
  training_set_scaled.shape
  
  print("\n==================================================")
  print("Skenario 1 -> bobot = 0; seq length = 2; hidden size = 3; input size = 3; output = 1")

  X_train = []
  y_train = []
  timestep = 2
  for i in range(timestep, training_set_scaled.shape[0]) :
      X_train.append(training_set_scaled[i-timestep:i, 0])
      y_train.append(training_set_scaled[i, 0])

  X_train, y_train = np.array(X_train)[:3], np.array(y_train)[:3]
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

  model_zero = SequentialModel([
    SimpleRNN(3, input_shape=(X_train.shape[1], 1), random_mode=0),
    Dense(1, "linear")
  ])

  out = model_zero.forward_propagation(X_train, y_train)
  predicted = scaler.inverse_transform(np.array(out))
  
  print("\nInversed predicted data")
  for t in range(len(predicted)):
    print("Sequence " + str(t+1) + ": " + str(predicted[t]))

  print("\n==================================================")
  print("Skenario 2 -> bobot = 1; seq length = 3; hidden size = 4; input size = 4; output = 1")

  X_train = []
  y_train = []
  timestep = 3
  for i in range(timestep, training_set_scaled.shape[0]) :
      X_train.append(training_set_scaled[i-timestep:i, 0])
      y_train.append(training_set_scaled[i, 0])

  X_train, y_train = np.array(X_train)[:4], np.array(y_train)[:4]
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

  model_one = SequentialModel([
    SimpleRNN(4, input_shape=(X_train.shape[1], 1), random_mode=1),
    Dense(1, "linear")
  ])

  out = model_zero.forward_propagation(X_train, y_train)
  predicted = scaler.inverse_transform(np.array(out))

  print("\nInversed predicted data")
  for t in range(len(predicted)):
    print("Sequence " + str(t+1) + ": " + str(predicted[t]))

  print("\n==================================================")
  print("Skenario 3 -> bobot = 1; seq length = 3; hidden size = 4; input size = 5; output = 1")

  X_train = []
  y_train = []
  timestep = 3
  for i in range(timestep, training_set_scaled.shape[0]) :
      X_train.append(training_set_scaled[i-timestep:i, 0])
      y_train.append(training_set_scaled[i, 0])

  X_train, y_train = np.array(X_train)[:5], np.array(y_train)[:5]
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

  model_random = SequentialModel([
    SimpleRNN(4, input_shape=(X_train.shape[1], 1), random_mode=2),
    Dense(1, "linear")
  ])

  out = model_random.forward_propagation(X_train, y_train)
  predicted = scaler.inverse_transform(np.array(out))
  
  print("\nInversed predicted data")
  for t in range(len(predicted)):
    print("Sequence " + str(t+1) + ": " + str(predicted[t]))