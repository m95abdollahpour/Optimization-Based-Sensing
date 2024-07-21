# deeper cnn model for mnist
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
import tensorflow as tf
from scipy import stats
import numpy as np
from scipy.sparse import random
import matplotlib.pyplot as plt
import pygad




# load train and test dataset
def load_dataset():
	# load dataset
    global testY
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainx = []
    rvs = stats.poisson(0, loc=1).rvs
    S = random(28, 28, density=0.2, format = 'csc', data_rvs = rvs)
    mask = S.A
    for i in range(len(trainX)):
        S = random(28, 28, density=.05, format = 'csc', data_rvs = rvs)
        mask = S.A
        sensed = trainX[i]*mask
        pos = np.where(mask != 0)
        sensed1 = sensed[pos]
        trainx.append(sensed1)

    testx = []
    for i in range(len(testX)):
        S = random(28, 28, density=.05, format = 'csc', data_rvs = rvs)
        mask = S.A
        sensed = testX[i]*mask
        pos = np.where(mask != 0)
        sensed1 = sensed[pos]
        testx.append(sensed1)

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return np.array(trainx), trainY, np.array(testx), testY

# scale pixels
def prep_pixels(train, test):

    train = train / 255.0
    test = test / 255.0
    return train, test

# define cnn model
def define_model():
    input_shape = (157, 1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters= 32, kernel_size=3, activation='relu',padding='same',input_shape= input_shape))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3,padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3,padding='same',activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# evaluate a model using k-fold cross-validation
def evaluate_model(trainX, trainY, testX, testY,n_folds=1):

    scores, histories = list(), list()
    global model
    model = define_model()
    history = model.fit(trainX, trainY, epochs=15, batch_size=1028, validation_data=(testX, testY), verbose=1)
    return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    plt.show()

# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()


# load dataset
trainX, trainY, testX, testY = load_dataset()
# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)
# evaluate model
scores, histories = evaluate_model(trainX, trainY, testX, testY)





(trainX, trainY), (testX, testY) = mnist.load_data()
trainX, testX = prep_pixels(trainX, testX)
testY = to_categorical(testY)


iii = 0

def fitness_func(solution, solution_idx):


    global iii
    global mask
    mask = solution.copy()
    mask = np.reshape(mask, (28,28))
    global sensed
    sensed = []

    for i in range(len(testX)):
        sensed.append(testX[i,:,:]*mask)


    sensed = np.array(sensed)
    sensed = sensed.reshape((sensed.shape[0], sensed.shape[1], sensed.shape[2], 1))
    _, acc = model.evaluate(sensed[:1000], testY[:1000], verbose=0)

    non_zero = len(np.where(mask==1)[0])

    if non_zero > 43:
        acc /= non_zero

    print(acc)
    iii += 1
    print(iii)
    return acc




img_size = 28*28
sparsity = 40
nonsparse = img_size - sparsity

gene_space = img_size * [[0, 1]]

initial_population = sparsity*[1]
initial_population.extend(nonsparse*[0])
initial_population = np.array(initial_population).reshape(len(initial_population), 1)
np.random.shuffle(initial_population)
initial_population = 10*[initial_population]
initial_population = np.array(initial_population)[:,:,0]

ga_instance = pygad.GA(num_generations=1000,
                       num_parents_mating=10,
                       initial_population = initial_population,
                       fitness_func=fitness_func,
                       init_range_low=0.0,
                       init_range_high=1.0,
                       gene_space=gene_space,
                       keep_elitism = 5,
                       crossover_type="single_point",
                       mutation_percent_genes=0.1,
                       mutation_type="random",
                       mutation_by_replacement=True,
                       random_mutation_min_val=0.0,
                       random_mutation_max_val=1 #,callback_generation=callback
                       )



ga_instance.run()









