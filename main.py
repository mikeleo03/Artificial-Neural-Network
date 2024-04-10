import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lib.file_io.file_handling import *
from lib.Sequential import Sequential
from lib.Layer import Dense

if __name__ == "__main__":
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    X_train, X_val, y_train, y_val = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)
    
    model = Sequential()
    model.add(Dense(4, activation='sigmoid', ))
    
