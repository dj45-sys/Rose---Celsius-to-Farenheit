#Imports :3

import tensorflow as tf
import numpy as np

#Data to study for  the model

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
farenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#Create a linear model

layer = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer])


model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error',
)

#Recieve the data and train the model

print("Starting training...")
history = model.fit(celsius,farenheit, epochs=1000, verbose=False)
print("Training finished.")

#Testing

import matplotlib.pyplot as plt

plt.xlabel("# Epoch")
plt.ylabel("Loss")
plt.plot(history.history['loss'])

#Recieve one data of float type

input = float(input("Enter a temperature in Celsius: "))

print("Starting prediction...")

#Calculate the prediction

result = model.predict([input])

#Print the prediction

print("Result is: "+ str(result) +" farenheit\n")

print("Internal Variables")
print(layer.get_weights())