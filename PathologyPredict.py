import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = .95
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

from keras import backend as K
K.set_image_dim_ordering('th')  # a lot of old examples of CNNs
# have the format (colors, rows, columns)  (1,28,28)
#  tensorflow expects (28,28,1)

img_rows = 200
img_cols = 200

import glob
allImageFileNames = glob.glob("/media/hdd0/unraiddisk2/cbis-ddsm/DOI/*/*/*/*200x200.png")
import numpy as np
import pandas as pd
calcification = pd.read_csv("/media/hdd0/unraiddisk2/cbis-ddsm/calcification.csv").values
masses = pd.read_csv("/media/hdd0/unraiddisk2/cbis-ddsm/masses.csv").values

def makeNumericalFromString(y):
 y_num = np.zeros(y.shape[0])
 for i in range(y.shape[0]):
   if y[i,11] == "MALIGNANT":
     y_num [i] = 0
   elif y[i,11] == "BENIGN":
     y_num [i] = 1
   else:
     y_num [i] = 2
 return y_num 

y_calc_num = makeNumericalFromString(calcification)
y_mass_num = makeNumericalFromString(masses)

y_num = np.append(y_calc_num,y_mass_num)
from keras.utils import np_utils
y = np_utils.to_categorical(y_num )

from scipy.misc import imread, imsave, imresize

x = np.zeros( (calcification.shape[0] + masses.shape[0] , img_rows, img_cols))
x_index = 0
for i in range(calcification.shape[0]):
 filename = calcification[i,0]
 print str(x_index)+" Loading "+filename
 x[x_index] = imresize( imread(filename, True) , (img_rows, img_cols) ).astype('float32') / 255
 x_index += 1

for i in range(masses.shape[0]):
 filename = masses[i,0]
 print str(x_index)+" Loading "+filename
 x[x_index] = imresize( imread(filename, True) , (img_rows, img_cols) ).astype('float32') / 255
 x_index += 1

x = x[:,np.newaxis,:,:]

nb_classes = 3

from keras.models import Sequential,model_from_json
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D, MaxPooling2D

model = Sequential()
nb_filters = 8  # number of convolutional filters to use
nb_pool = 2     # size of pooling area for max pooling
nb_conv = 3     # convolution kernel size
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols), activation='relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv, activation='relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, batch_size=256, nb_epoch=12, verbose=1) 
#model.fit(x, y, validation_data=(x_test, y_test), batch_size=256, nb_epoch=12, verbose=1) 

from sklearn.cross_validation import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(x,y,test_size=0.2)
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=256, nb_epoch=36, verbose=1) 
