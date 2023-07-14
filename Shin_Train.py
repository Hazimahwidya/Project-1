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
