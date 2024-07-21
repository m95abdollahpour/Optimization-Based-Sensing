from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from scipy import stats
import numpy as np
from scipy.sparse import random
import matplotlib.pyplot as plt
import random as random2
import pygad
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''





(trainX, trainY), (testX, testY) = mnist.load_data()


rvs = stats.poisson(0, loc=1).rvs
for i in range(len(trainX)):
    S = random(28, 28, density=0.05, format = 'csc', data_rvs = rvs)
    mask = S.A
    trainX[i] = trainX[i]*mask


for i in range(len(testX)):
    S = random(28, 28, density=0.05, format = 'csc', data_rvs = rvs)
    mask = S.A
    testX[i] = testX[i]*mask

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))
# one hot encode target values
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# scale pixels
def prep_pixels(train, test):
    train = train / 255.0
    test = test / 255.0
    # return normalized images
    return train, test

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(trainX, trainY, testX, testY,n_folds=1):

    scores, histories = list(), list()

    # define model
    global model
    model = define_model()
    # select rows for train and test

    history = model.fit(trainX, trainY, epochs=65, batch_size=1028, validation_data=(testX, testY), verbose=1)

    return scores, histories



# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()

# summarize model performance
def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()



trainX, testX = prep_pixels(trainX, testX)
# evaluate model
scores, histories = evaluate_model(trainX, trainY, testX, testY)
# learning curves
summarize_diagnostics(histories)


(trainX, trainY), (testX, testY) = mnist.load_data()
trainX, testX = prep_pixels(trainX, testX)
testY = to_categorical(testY)
trainY = to_categorical(trainY)

fit_data = trainX[:5000]
fit_data_labels = trainY[:5000]

iii = 0
acc_tr = []
acc_te = []
def fitness_func(solution, solution_idx):


    global iii
    global mask
    global mask_pos
    mask_pos = solution.copy()
    mask = np.zeros((28*28))
    mask[np.uint(mask_pos)] = 1
    mask = np.reshape(mask, (28,28))
    global sensed

    sensed = fit_data.copy()
    sensed = np.multiply(sensed, mask)

    sensed = sensed.reshape((sensed.shape[0], 28, 28, 1))

    sensed = np.array(sensed)
    loss, acc = model.evaluate(sensed, fit_data_labels, verbose=0)
    acc_tr.append(acc)

    global sensed2
    sensed2 = testX.copy()
    sensed2 = np.multiply(sensed2, mask)
    sensed2 = sensed2.reshape((sensed2.shape[0], 28, 28, 1))
    sensed2 = np.array(sensed2)
    loss, acc2 = model.evaluate(sensed2, testY, verbose=0)
    acc_te.append(acc2)

    print(acc)
    iii += 1
    print(iii)
    return acc




img_size = 28*28
sparsity = 39
nonsparse = img_size - sparsity
gene_space = range(img_size-1)


# the variable size should be defined in init population
initial_population = random2.sample(range(img_size-1), sparsity)
initial_population = np.array(initial_population).reshape(len(initial_population), 1)
initial_population = 10*[initial_population]
initial_population = np.array(initial_population)[:,:,0]


ga_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=10,
                       initial_population = initial_population,
                       fitness_func=fitness_func,
                       gene_space=gene_space,
                       keep_elitism = 5,
                       crossover_type="single_point",
                       mutation_percent_genes=0.1,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=-1,
                       random_mutation_max_val=1 #,callback_generation=callback
                       )



ga_instance.run()


acc_te = np.array(acc_te)
acc_tr = np.array(acc_tr)
