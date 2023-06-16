import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
from typing import Tuple
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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
print(accuracy_score(test_labels , model0.predict(test_data)) , accuracy_score(train_labels , model0.predict(train_data)))
print("\n",confusion_matrix(test_labels, model0.predict(test_data)))

# neural network which is responsible for feature selection --> outputs feature numbers with the most impact on model

def training(train_data, train_labels,test_data, test_labels ,index):
    input_tensor = tf.keras.Input(shape=(index.shape[0],))
    hidden_size = 2048
    reg_l1_param = 1.05e-4

    hidden_layer_1 = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.relu,
                                            name = 'hidden_layer',
                                            activity_regularizer=tf.keras.regularizers.l1(reg_l1_param))(input_tensor)

    output_layer = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, name = 'classification_layer')(hidden_layer_1)

    model = tf.keras.Model(inputs=input_tensor, outputs=output_layer)
    model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels, epochs=10, batch_size=32 , validation_data=(test_data, test_labels))
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    arg_max1 = np.argmax(np.mean(model.get_layer('classification_layer').get_weights()[0] , axis = 1))
    x = np.sort(model.get_layer("hidden_layer").get_weights()[0][:,arg_max1])[::-1]
    captured = index[np.squeeze(np.array([np.argwhere(model.get_layer("hidden_layer").get_weights()[0][:,arg_max1] == i) for i in x[:25]]))]

    return captured

# choose the best 25 features on each try

index = np.arange(1024)
g_features = list()
captured = list()
for i in range(10):
    index = np.setdiff1d(index , captured)
    captured = training(train_data[:,index], train_labels ,test_data[:,index], test_labels , index)
    g_features.extend(captured)

# train model with obtained features

x = 25
results = list()
print("Score" , "    Accuracy","    Features" , "\n")
for i in range(10):
  model3 = SVC(C = 2.2, gamma = 'scale' , kernel = 'rbf')
  model3.fit(tf.squeeze(train_data[:,g_features[:((i+1)*x)]]), train_labels)
  accuracy = accuracy_score(test_labels , model3.predict(tf.squeeze(test_data[:,g_features[:((i+1)*x)]])))
  score = accuracy - ((i+1) * x - x ) * 0.00075
  print("{:.4f}".format(np.round(score , 4)) ,"    ", "{:.3f}".format(np.round(accuracy , 3)) ,"      ", (i+1)*x)
  results.append(((i+1)*x,np.round(score , 4) , np.round(accuracy , 3)))