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
# (1) dataset and model
#  ----------------
num_inp, num_tgt = 11, 4

dataframe_train = pd.read_csv('ShinTrainOneHot_dataset.csv', delimiter=',')
dataset_train = dataframe_train.to_numpy().astype('float32')

dataframe_test = pd.read_csv('ShinTestOneHot_dataset.csv', delimiter=',')
dataset_test = dataframe_test.to_numpy().astype('float32')

model = tf.keras.models.load_model('MeatQualityIdentShin_Softmax_10.h5')

# (2) split into training/validation set
#  ----------------
inpvalid = dataset_train[:, 0:num_inp]
truevalid = dataset_train[:, num_inp:]
inptest = dataset_test[:, 0:num_inp]
truetest = dataset_test[:, num_inp:]
# (3) inference
#  ----------------
predvalid = model.predict(inpvalid/60)
predtest = model.predict(inptest/60)
predvalidround = np.round(predvalid)
predtestround = np.round(predtest)

# (4) accuracy
#  ----------------
print('--- Result ---')
accvalid = accuracy_score((truevalid), (predvalidround))
print('> Validation Accuracy (%) : ', np.round(accvalid * 100, 2))
acctest = accuracy_score((truetest), (predtestround))
print('> Test Accuracy (%)       : ', np.round(acctest * 100, 2))

# (5) confusion matrix
#  ----------------
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
