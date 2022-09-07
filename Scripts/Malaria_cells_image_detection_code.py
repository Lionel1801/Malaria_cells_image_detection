# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 00:14:08 2022

@author: Alfa
"""


### Inhalte

# Motivation
# Datensatz beschreiben
# Code erklären
# Visualisierungen erklären
# Interpretation
# Fazit


### Motivation

# Die Problematik Malaria, ist immernoch ein Thema in Tropische und Sud Sahara Länder.
# Malaria ist ein wichtiger Faktor für die höhe Sterberate in Afrika beispielweise.
# Jede effektive Methode Malaria infizierte Zellen zu identifizieren, hilft bei der Bekämpfung
# Der Datensatz hat 27558 cell_images, geteilt in 'Parasitized' und 'Uninfected'.
# Die Klassifikation der Bilder mit  CNN, 
# 





# Import Libraries

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import os
import PIL
import pathlib
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
from matplotlib.pyplot import figure




data = pathlib.Path('C:/Users/alfa/Desktop/Deep Learning/Projektarbeit/cell_images')  # Naming the path of the dataset



image_count = len(list(data.glob('*/*.png')))  # Counting the dataset
print(image_count)


parasitized = list(data.glob('Parasitized/*'))      # Visualizing the morphology of a parasitized cell
PIL.Image.open(str(parasitized[0]))

uninfected = list(data.glob('Uninfected/*'))        #Visualizing the morphology of an uninfected cell
PIL.Image.open(str(uninfected[0]))



# Data Loading
# Creating Dataset

batch_size = 200
img_height = 75
img_width = 75

train_ds = tf.keras.utils.image_dataset_from_directory(
  data,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)




val_ds = tf.keras.utils.image_dataset_from_directory(
  data,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)



# Defining the model architecture

model = keras.Sequential()

model.add(keras.layers.Conv2D(16, kernel_size=3,strides=1, activation=None, 
                              input_shape=(75,75,3)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(32, kernel_size=3,strides=1, activation=None))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv2D(64, kernel_size=3,strides=1, activation=None))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Conv2D(32, kernel_size=3,strides=1, activation=None))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv2D(16, kernel_size=3,strides=1, activation=None))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Conv2D(8, kernel_size=3,strides=1, activation=None))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(units=1))
model.add(keras.layers.Activation('sigmoid'))


# Summary of the model


model.summary()


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



# Training the model on training dataset and validating it on validation data

history = model.fit(train_ds,
                   epochs=20,
                   validation_data=val_ds,
                   batch_size=batch_size)


# Visualizing the Learning Curve


def plotLearningCurve(history,epochs):
  epochRange = range(1,epochs+1)
  plt.plot(epochRange,history.history['accuracy'])
  plt.plot(epochRange,history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()

  plt.plot(epochRange,history.history['loss'])
  plt.plot(epochRange,history.history['val_loss'])
  plt.title('Model Loss') 
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train','Validation'],loc='upper left')
  plt.show()

plotLearningCurve(history,20)


# Confusion Matrix

      
all_labels = []
all_y_preds = []

for images, labels in val_ds:
    print('images.shape:', images.shape)
    print('labels.shape:', labels.shape)

    y_test_pred = model.predict(images)
    all_labels.extend(labels)
    all_y_preds.extend(y_test_pred)

plt.scatter(all_labels, all_y_preds)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.show()


y_test_pred = model.predict(val_ds)
y_test = np.concatenate([y for x, y in val_ds], axis=0)

y_test_pred_class = np.round(all_y_preds).reshape(-1,1)
y_test_class      = np.round(all_labels).reshape(-1,1)
  


# Creates a confusion matrix
cm = confusion_matrix(y_test_class, y_test_pred_class, normalize='true') 

# Transform to df for easier plotting
cm_df = pd.DataFrame(cm,
                      index = ['Parasitized','Uninfected'], 
                      columns = ['Parasitized','Uninfected'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()




## Interpretation

# Aus der Confusion Matrix, kann man sehen wie TP, FP, TN, FN liegen

# TP    FN
# FP    TN

# Die Precision kann ausgerechnet werden: Precision = TP/(TP+FP)
# => 0.95/(0.95+0.027) = 0.9724
# Entspricht ein Precision von 97.3%

# Der Recall kann auch ausgerechnet werden: Recall = TP/(TP+FN) 
# => 0.95/(0.95+0.054) = 0.9462
# Entspricht ein Recall von 94.6%




# Fazit 

#Ziel der vorliegenden Projekt war Malaria infizierte Zellen von nicht infizierte 
# Zellen zu unterscheiden/erkennen.
# Die Ergebnisse des Projekts zeigen, dass CNN ein sehr guter Model ist für Image 
# Classificaton.
# Die Convolutional und Pooling Layers helfen bei der Preprocessing
# Die Ergebnisse der Binary Image Classification: accuracy von 0.9609, val_loss: 0.1333, val_accuracy : 0.9574
# Entspricht ein Prozentsatz von 95.7%
# Ein Vergleich mit anderen Modellen ist mit der erzielten 95.7%, nicht nötig.
# Außerdem haben wir leider ein Mangel an Hardware Memory Leistung.
# Versuche mit weiteren Modellen hätte größe Einfluss auf unseren Zeitmanagement
# gehabt.















