import csv
import numpy as np
import os
import pickle
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

labels = np.random.randint(2, size=(1000, 1))
print("end")