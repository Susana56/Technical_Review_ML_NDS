import torch
import numpy as np
import pandas as pd
from numpy import array
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import matplotlib.pyplot as plt


def fx(x, y, z, t): return sigma*(y-x)
def fy(x, y, z, t): return x*(rho-z)-y
def fz(x, y, z, t): return x*y-beta*z

#a) Defining the classical Runge-Kutta 4th order method

def RungeKutta45(x,y,z,fx,fy,fz,t,h):
    k1x, k1y, k1z = (h*f(x, y, z, t) for f in (fx, fy, fz) )
    xs, ys, zs, ts = (r+0.5*kr for r, kr in zip((x, y, z, t), (k1x, k1y, k1z, h)))
    k2x, k2y, k2z = (h*f(xs, ys, zs, ts) for f in (fx, fy, fz))
    xs, ys, zs, ts = (r+0.5*kr for r, kr in zip((x, y, z, t), (k2x, k2y, k2z, h)))
    k3x, k3y, k3z = (h*f(xs, ys, zs, ts) for f in (fx, fy, fz))
    xs, ys, zs, ts = (r+kr for r, kr in zip((x, y, z, t), (k3x, k3y, k3z, h)))
    k4x, k4y, k4z = (h*f(xs, ys, zs, ts) for f in (fx, fy, fz))
    return (r+(k1r+2*k2r+2*k3r+k4r)/6 for r, k1r, k2r, k3r, k4r in
            zip((x, y, z), (k1x, k1y, k1z), (k2x, k2y, k2z), (k3x, k3y, k3z), (k4x, k4y, k4z)))


sigma = 10.
beta = 8./3.
rho = 28.  # same parameters as the paper
tIn = 0.
tFin = 99.
h = 0.01
totalSteps = int(np.floor((tFin-tIn)/h))

t = totalSteps * [0.0]
x = totalSteps * [0.0]
y = totalSteps * [0.0]
z = totalSteps * [0.0]


x[0], y[0], z[0], t[0] = 1., 1., 1., 0.  #Initial condition
for i in range(1, totalSteps):
    t[i] = t[i-1]+h
    x[i], y[i], z[i] = RungeKutta45(x[i-1], y[i-1], z[i-1], fx, fy, fz, t[i-1], h)


#Producing reconstructed time series

sequence_length = 20
x_train = x[0:6900]
dataset_tr_x = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_train, x_train, sequence_length, batch_size=1)
#print('Samples: %d' % len(dataset_te))  # prints the total number of sampled 6880

x_test = x[6900:]
dataset_te_x = tf.keras.preprocessing.sequence.TimeseriesGenerator(x_test, x_test, sequence_length, batch_size=1)

y_train = y[0:6900]
dataset_tr_y = tf.keras.preprocessing.sequence.TimeseriesGenerator(y_train, y_train, sequence_length, batch_size=1)

y_test = y[6900:]
dataset_te_y = tf.keras.preprocessing.sequence.TimeseriesGenerator(y_test, y_test, sequence_length, batch_size=1)

z_train = z[0:6900]
dataset_tr_z = tf.keras.preprocessing.sequence.TimeseriesGenerator(z_train, z_train, sequence_length, batch_size=1)

z_test = z[6900:]
dataset_te_z = tf.keras.preprocessing.sequence.TimeseriesGenerator(z_test, z_test, sequence_length, batch_size=1)



model_x = keras.Sequential()
model_x.add(layers.Dense(100, activation='relu', input_dim=sequence_length))
model_x.add(layers.Dense(10, activation='relu'))
model_x.add(layers.Dense(1))

opt = keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9)
model_x.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# fit model
model_x.fit(dataset_tr_x, steps_per_epoch=6880, epochs=10, verbose=2)

model_x.evaluate(dataset_te_x, verbose=2)

model_y = keras.Sequential()
model_y.add(layers.Dense(100, activation='relu', input_dim=sequence_length))
model_y.add(layers.Dense(10, activation='relu'))
model_y.add(layers.Dense(1))

opt = keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9)
model_y.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# fit model
model_y.fit(dataset_tr_y, steps_per_epoch=6880, epochs=10, verbose=2)

model_y.evaluate(dataset_te_y, verbose=2)

model_z = keras.Sequential()
model_z.add(layers.Dense(100, activation='relu', input_dim=sequence_length))
model_z.add(layers.Dense(10, activation='relu'))
model_z.add(layers.Dense(1))

opt = keras.optimizers.SGD(learning_rate=0.000001, momentum=0.9)
model_z.compile(optimizer=opt, loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# fit model
model_z.fit(dataset_tr_z, steps_per_epoch=6880, epochs=10, verbose=2)

model_z.evaluate(dataset_te_z, verbose=2)


y_pred_x = [None]*len(dataset_te_x)
y_pred_y = [None]*len(dataset_te_y)
y_pred_z = [None]*len(dataset_te_z)
for i in range(len(dataset_te_x)):  # prints the reconstructed input/output pairs
    x_temp_x, y_x = dataset_te_x[i]
    x_temp_y, y_y = dataset_te_y[i]
    x_temp_z, y_z = dataset_te_z[i]
    ynew_x = model_x.predict(x_temp_x)
    ynew_y = model_y.predict(x_temp_y)
    ynew_z = model_z.predict(x_temp_z)
    y_pred_x[i] = ynew_x[0, 0]
    y_pred_y[i] = ynew_y[0, 0]
    y_pred_z[i] = ynew_z[0, 0]

#Rewriting code for to use the output within the input
y_pred_new_x = [None] * len(dataset_te_x)
y_pred_new_y = [None] * len(dataset_te_y)
y_pred_new_z = [None] * len(dataset_te_z)
x_start_x, y_x = dataset_te_x[0]
x_start_y, y_y = dataset_te_y[0]
x_start_z, y_z = dataset_te_z[0]
for i in range(len(dataset_te_x)):
    if i == 0:
        x_temp_x = x_start_x
        x_temp_y = x_start_y
        x_temp_z = x_start_z
    else:
        x_place_x = x_temp_x
        x_place_y = x_temp_y
        x_place_z = x_temp_z
        for j in range(19):   # values 0 to 18, missing last index
            x_temp_x[0, j] = x_place_x[0, j+1]
            x_temp_y[0, j] = x_place_y[0, j + 1]
            x_temp_z[0, j] = x_place_z[0, j + 1]
        x_temp_x[0, 19] = y_pred_new_x[i-1]           # newest value, get predicted value from last output
        x_temp_y[0, 19] = y_pred_new_y[i - 1]  # newest value, get predicted value from last output
        x_temp_z[0, 19] = y_pred_new_z[i - 1]  # newest value, get predicted value from last output
    ynew_x = model_x.predict(x_temp_x)
    ynew_y = model_y.predict(x_temp_y)
    ynew_z = model_z.predict(x_temp_z)
    y_pred_new_x[i] = ynew_x[0, 0]
    y_pred_new_y[i] = ynew_y[0, 0]
    y_pred_new_z[i] = ynew_z[0, 0]

t_new = t[6920:9900]
x_original = x[6920:9900]
y_original = y[6920:9900]
z_original = z[6920:9900]

df = pd.DataFrame({"t_new": t_new, "x_original": x_original, "y_original": y_original, "z_original": z_original,
                   "y_pred_x": y_pred_x, "y_pred_y": y_pred_y, "y_pred_z": y_pred_z, "y_pred_new_x": y_pred_new_x,
                   "y_pred_new_y": y_pred_new_y, "y_pred_new_z": y_pred_new_z})
df.to_csv("Lorenz_x_new")
