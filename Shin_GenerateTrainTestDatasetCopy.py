# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k5dfKfzeS2KpR0ahIb-gFqvuhFITqs3l
"""

import pandas as pd
import numpy as np
path_ = "/content/drive/MyDrive/Colab Notebooks/Shin.csv"
dataset = pd.read_csv(path_)
dataset

dataset.head()



# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

X = dataset[['MQ135','MQ136','MQ137','MQ138','MQ2','MQ3','MQ4','MQ5','MQ6','MQ8','MQ9','Label']]
Y = dataset['Minute']

X

Y

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state = 0)

len(X_train)

len(X_test)

X_train

X_train.to_csv('ShinTrain_dataset.csv', index=False)

X_test

X_test.to_csv('ShinTest_dataset.csv', index=False)

dataset.info

y_train

y_test

import pandas as pd
import numpy as np
path_ = "/content/ShinTrain_dataset.csv"
dataset = pd.read_csv(path_)
dataset

from sklearn.preprocessing import OneHotEncoder

one_hot_encoded_train = pd.get_dummies(dataset, columns = ['Label'])
print(one_hot_encoded_train)

# save concatenated data to CSV
one_hot_encoded_train.to_csv('ShinTrainOneHot_dataset.csv', index=False)

import pandas as pd
import numpy as np
path_ = "/content/ShinTest_dataset.csv"
dataset = pd.read_csv(path_)
dataset

from sklearn.preprocessing import OneHotEncoder

one_hot_encoded_test = pd.get_dummies(dataset, columns = ['Label'])
print(one_hot_encoded_test)

# save concatenated data to CSV
one_hot_encoded_test.to_csv('ShinTestOneHot_dataset.csv', index=False)

import pandas as pd
import numpy as np
path_ = "/content/Inside-OutsideTrain_dataset.csv"
dataset = pd.read_csv(path_)
dataset

from sklearn.preprocessing import OneHotEncoder

one_hot_encoded_data = pd.get_dummies(dataset, columns = ['Label'])
print(one_hot_encoded_data)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import pandas as pd
import datetime
import random
from sklearn.metrics import accuracy_score

# --- Main Program ---
#=====================

# (1) Load the data
#-------------------
num_inp, num_tgt = 11, 4

dataframe_train = pd.read_csv('ShinTrainOneHot_dataset.csv', delimiter=',')
dataset_train = dataframe_train.to_numpy().astype('float32')

# (2) dataset to train variable
#------------------------------

inp = dataset_train[:, 0:num_inp]
tgt = dataset_train[:, num_inp:]

# (3) normalize training data
#----------------------------
inpnorm = inp / 60

# (4) build the model
#---------------------
num_neuron_hl = 10
inputs = tf.keras.Input(shape=(num_inp))
x = inputs
x = tf.keras.layers.Dense(num_neuron_hl, activation='sigmoid')(x)
x = tf.keras.layers.Dense(num_tgt, activation='softmax')(x)
outputs = x

model = tf.keras.Model(inputs, outputs)
model.summary()

# (5) training model
#--------------------
num_epoch = 5000; lr = 0.005; batch_size = 32
ismodelsaved = True
model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
start_time = time.time()
model.fit(inpnorm, tgt, epochs=num_epoch, batch_size=batch_size)
stop_time = time.time() - start_time
training_time = datetime.timedelta(seconds=stop_time)
print('Training time: ', training_time, '\n')

if ismodelsaved:
    model.save('MeatQualityIdentShin_Softmax_'+str(num_neuron_hl)+'.h5')
    print('Model saved\n')

# (6) validation
#----------------
predvalid = model.predict(inpnorm)
predvalidround = np.round(predvalid)
accvalid = accuracy_score(tgt, predvalidround)
print('Validation accuracy (%) : ', np.round(accvalid*100, 2), '\n')

# (7) test
#---------
dataframe_test = pd.read_csv('ShinTestOneHot_dataset.csv', delimiter=',')
dataset_test = dataframe_test.to_numpy().astype('float32')

inptest = dataset_test[:, 0:num_inp]
tgttest = dataset_test[:, num_inp:]

inptestnorm = inptest / 60
predtest = model.predict(inptestnorm)
predtestround = np.round(predtest)
acctest = accuracy_score(tgttest, predtestround)
print('Test accuracy (%) : ', np.round(acctest*100, 2), '\n')
print('End of program\n')

import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import pandas as pd
import datetime
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def onehot2label(onehot):
    return np.argmax(onehot, axis=1)

# --- Main Program ---
# ====================

# (1) --- dataset and model
num_inp, num_tgt = 11, 4

dataframe_train = pd.read_csv('ShinTrainOneHot_dataset.csv', delimiter=',')
dataset_train = dataframe_train.to_numpy().astype('float32')

dataframe_test = pd.read_csv('ShinTestOneHot_dataset.csv', delimiter=',')
dataset_test = dataframe_test.to_numpy().astype('float32')

model = tf.keras.models.load_model('MeatQualityIdentShin_Softmax_10.h5')

# (2) --- split into training/validation set

inpvalid = dataset_train[:, 0:num_inp]
truevalid = dataset_train[:, num_inp:]

inptest = dataset_test[:, 0:num_inp]
truetest = dataset_test[:, num_inp:]

# (3) --- inference
predvalid = model.predict(inpvalid/60)
predtest = model.predict(inptest/60)
predvalidround = np.round(predvalid)
predtestround = np.round(predtest)

# (4) --- accuracy
print('--- Result ---')
accvalid = accuracy_score((truevalid), (predvalidround))
print('> Validation Accuracy (%) : ', np.round(accvalid * 100, 2))
acctest = accuracy_score((truetest), (predtestround))
print('> Test Accuracy (%)       : ', np.round(acctest * 100, 2))

# (5) --- confusion matrix
# data validasi
truelabelvalid = onehot2label(truevalid)
predlabelvalid = onehot2label(predvalidround)
# data test
truelabeltest = onehot2label(truetest)
predlabeltest = onehot2label(predtestround)

cm_valid = metrics.confusion_matrix(truelabelvalid, predlabelvalid)
cm_test = metrics.confusion_matrix(truelabeltest, predlabeltest)
cm_valid_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_valid,
    display_labels=['excellent', 'good', 'acceptable', 'spoiled'])
cm_test_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_test,
    display_labels=['excellent', 'good', 'acceptable', 'spoiled'])
cm_valid_display.plot()
cm_test_display.plot()

plt.show()
print('\nEnd of Program\n')