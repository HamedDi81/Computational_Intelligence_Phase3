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