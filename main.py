import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
from typing import Tuple
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the data and label
def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(f'{filename}')
    return data['data'], data['labels']

# Split the data and labels into training and testing sets

train_data, train_labels = load_data('/content/drive/MyDrive/Copy of train_data_SYN.npz') # Train Data Path

print(train_data.shape)
print(np.unique(train_labels, return_counts=True))

test_data, test_labels = load_data('/content/drive/MyDrive/Copy of test_data_SYN.npz') # Test Data Path

print(test_data.shape)
print(np.unique(test_labels, return_counts=True)) 

model0 = SVC(C = 2.2 , gamma = 'scale' , kernel = 'rbf' , verbose = 1)
model0.fit(train_data , train_labels)
accuracy_score(test_labels , model0.predict(test_data)) , accuracy_score(train_labels , model0.predict(train_data))

# neural network which is responsible for feature selection --> outputs feature numbers with the most impact on model

def training(train_data, train_labels,test_data, test_labels , variance):
  input_tensor = tf.keras.Input(shape=(1024,))
  tf.random.set_seed(1234)
  hidden_size = 2048
  reg_l1_param = 1e-4

  hidden_layer_1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.relu, 
                                        name = 'hidden_layer',
                                        activity_regularizer=tf.keras.regularizers.l1(reg_l1_param))(input_tensor)

  output_layer = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, name = 'classification_layer')(hidden_layer_1)

  model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)
  model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  history = model.fit(train_data, train_labels, epochs=10, batch_size=32 , validation_data=(test_data, test_labels))
  test_loss, test_acc = model.evaluate(test_data, test_labels)
  arg_max1 = np.argmax(np.mean(abs(model.get_layer('classification_layer').get_weights()[0]) , axis = 1))
  thresh = np.mean(abs(model.get_layer("hidden_layer").get_weights()[0][:,arg_max1])) + variance
  lg2_arg = np.argwhere(model.get_layer("hidden_layer").get_weights()[0][:,arg_max1] > thresh)
  return lg2_arg

# select the best common features among different iterations --> returns the intersection of all best features

def return_common_elements(times = 1 , variance = - 0.05):
  res = []
  for i in range(times):
    x = training(train_data, train_labels,test_data, test_labels , variance)
    if i > 0:
      temp = res[0]
      res.pop()
      res.append(np.intersect1d(temp , x))
    else:
      res.append(x)
  return res[0]

x = return_common_elements(times = 4 , variance = -0.04)
print(x.shape)
