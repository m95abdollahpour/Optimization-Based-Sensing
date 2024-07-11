








from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import tensorflow
tensorflow.random.set_seed(73)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from scipy import stats
import numpy as np
from scipy.sparse import random
import matplotlib.pyplot as plt
import random as random2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
import pygad

import os





















(trainX, trainY), (testX, testY) = mnist.load_data()


fit_data = trainX[:5000].copy()
fit_data_labels = trainY[:5000].copy()




data = fit_data.copy()
Sensed = []
for i in range(len(data)):
    d = np.ravel(data[i])
    Sensed.append(d)
    
Train = np.array(Sensed)


test_data = trainX[5000:10000].copy()
test_data_labels = trainY[5000:10000].copy()


data = test_data.copy()
Sensed = []
for i in range(len(data)):
    d = np.ravel(data[i])
    Sensed.append(d)
    
Test = np.array(Sensed)




sampling_point = 8


iii = 0
def fitness_func(solution, solution_idx):


    global iii
    global phi
    phi = solution.copy()
    phi = np.reshape(phi, (sampling_point, 784))
    global sensed



    Sensed = []
    for i in range(len(Train)):
        Sensed.append(np.dot(phi, Train[i]/255))
        
    Sensed_tr = np.array(Sensed)




    Sensed2 = []
    for i in range(len(Test)):
        Sensed2.append(np.dot(phi, Test[i]/255))
        
    Sensed_te = np.array(Sensed2)



    verbose, epochs, batch_size = 0, 50, 5000
    n_timesteps, n_outputs = Sensed_tr.shape[1], 10
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    hist = model.fit(Sensed_tr, fit_data_labels, epochs=epochs, batch_size=batch_size, verbose=verbose)
    acc = np.mean(hist.history['accuracy'][-5:-1])

    iii += 1
    return acc





gene_space = sampling_point*784 * [[-1,0, 1]] #for binary

sensor_size = sampling_point*784



ga_instance = pygad.GA(num_generations=5,
                       num_parents_mating=25,
                    #    initial_population = initial_population,
                       fitness_func=fitness_func,
                       sol_per_pop=30,
                       num_genes=sensor_size,
                       init_range_low=0.0, #for binary
                       init_range_high=1.0, #for binary
                       gene_space=gene_space, #for binary
                       keep_elitism = 5,
                       crossover_type="single_point",
                       mutation_percent_genes=0.1,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1 #,callback_generation=callback
                       )



ga_instance.run()

np.save('phi_full_1-0-1_8_cnn_2.npy', phi)













